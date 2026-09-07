"""
Validation contract for federated-evaluation reports (PMCS).

This module does NOT assemble a report. Another piece of code builds the
actual nested dict with real Dice/HD95/communication numbers; this module
only provides:
  - the schema/accounting version strings embedded in every report,
  - naming helpers (partitioned_key, add_deprecated_alias),
  - the explicit list of required top-level fields,
  - validate_report(), which makes it structurally impossible to silently
    ship a report that is missing required fields or that contains NaN/Inf
    anywhere in its tree (a real past incident: one run's report shipped
    missing 4 required fields with no error).

Everything a report field holds is expected to already be plain,
JSON-safe Python (None/bool/str/int/float/dict/list/tuple) by the time it
reaches validate_report / is_finite_or_null -- the assembler is responsible
for converting numpy scalars/arrays and torch tensors beforehand. Passing a
raw numpy array or similar in here is a programming error, not a validation
failure, so is_finite_or_null raises TypeError for it rather than returning
False.
"""
import math

SCHEMA_VERSION = '2.0.0'
ACCOUNTING_VERSION = '2.0.0'

VALID_PARTITIONS = ('validation', 'test')
VALID_ROUND_SELECTIONS = ('last_round', 'best_validation')

# Explicit and importable (not buried in a function) so the assembler and
# any test can both import this list and extend it.
REQUIRED_TOP_LEVEL_FIELDS = [
    'schema_version',
    'accounting_version',
    'evaluation_partition',
    'round_selection',
    'communication',
    'encoder_coverage',
    'postproc_policy',
    'hd95_policy',
    'client_modalities',
    'provenance',
]

# Base names that must each appear as "{partition}_{basename}" (see
# partitioned_key), where `partition` is report['evaluation_partition'].
# These carry the actual per-region numbers -- a report can satisfy every
# name in REQUIRED_TOP_LEVEL_FIELDS (all metadata/bookkeeping) while still
# being missing the Dice/HD95 matrices themselves, which is exactly the
# hole this closes.
REQUIRED_PARTITION_MATRIX_BASENAMES = [
    'dice_matrix',
    'dice_postproc_matrix',
    'hd95_matrix',
    'hd95_postproc_matrix',
    'hd95_valid_pairs_only_matrix',
    'personalisation_gain_matrix',
]

# Additional top-level fields required regardless of partition: the
# canonical global-model block, per-case distributional stats, the
# cross-client fairness summary, and evidence that residual zeroing actually
# happened (see the residual_zeroing_log / global_model checks below).
REQUIRED_ADDITIONAL_FIELDS = [
    'global_model',
    'per_case_stats',
    'fairness',
    'residual_zeroing_log',
]

REQUIRED_GLOBAL_MODEL_FIELDS = ('dice', 'per_region', 'n_cases', 'partition', 'private_residual_zeroed')


class ReportValidationError(Exception):
    """Raised by validate_report() with every problem found, not just the first."""
    pass


def partitioned_key(base_name: str, partition: str) -> str:
    """e.g. partitioned_key('dice_matrix', 'test') -> 'test_dice_matrix'."""
    if partition not in VALID_PARTITIONS:
        raise ValueError(
            "partition must be one of %r, got %r" % (VALID_PARTITIONS, partition)
        )
    return '%s_%s' % (partition, base_name)


def is_finite_or_null(value) -> bool:
    """Recursively check a JSON-like value for NaN/Inf.

    A value passes if it is None, a bool/str/int, a finite float, or a
    dict/list/tuple all of whose elements recursively pass. Returns False
    for a NaN/+-inf float. Raises TypeError for any other type (e.g. a raw
    numpy array) -- the caller must have already converted everything to
    plain Python/JSON-safe types before validation.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(is_finite_or_null(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite_or_null(v) for v in value)
    raise TypeError(
        "is_finite_or_null: unexpected type %r (value must already be "
        "plain Python/JSON-safe; convert numpy/torch types before "
        "validation)" % (type(value),)
    )


def _find_non_finite_paths(value, path):
    """Walk value, returning a list of dotted-path strings for every
    NaN/+-inf float found. path is the dotted-path prefix so far ('' at
    the root)."""
    problems = []
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(path if path else '<root>')
    elif isinstance(value, dict):
        for k, v in value.items():
            child_path = '%s.%s' % (path, k) if path else str(k)
            problems.extend(_find_non_finite_paths(v, child_path))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            child_path = '%s.%s' % (path, i) if path else str(i)
            problems.extend(_find_non_finite_paths(v, child_path))
    # None/bool/str/int and anything else: nothing to recurse into.
    return problems


def validate_report(report: dict) -> None:
    """Raise ReportValidationError listing ALL problems found, or return None.

    Checks:
      (a) every name in REQUIRED_TOP_LEVEL_FIELDS is present in report;
      (b) report['evaluation_partition'] is one of VALID_PARTITIONS;
      (c) report['round_selection'] is a dict with 'mode' in
          VALID_ROUND_SELECTIONS and an int 'round_index';
      (d) every name in REQUIRED_PARTITION_MATRIX_BASENAMES is present under
          its "{partition}_" key -- a report can carry every metadata field
          and still be missing the actual Dice/HD95 numbers, which is a
          structurally incomplete report just as much as a missing metadata
          field is;
      (e) every name in REQUIRED_ADDITIONAL_FIELDS is present, and
          'global_model'/'residual_zeroing_log' additionally carry real
          evidence (not just the key existing): global_model has all of
          REQUIRED_GLOBAL_MODEL_FIELDS and private_residual_zeroed is
          literally True; residual_zeroing_log is a non-empty mapping whose
          every value is itself a non-empty list of zeroed tensor names (a
          report where zeroing silently matched nothing must fail here even
          if evaluate.py's own runtime assertion were ever bypassed);
      (f) no NaN/+-inf anywhere in the tree (dotted-path locations of every
          offender are collected into the error message).
    """
    problems = []

    missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in report]
    if missing:
        problems.append(
            "missing required top-level field(s): %s" % ', '.join(missing)
        )

    missing_additional = [f for f in REQUIRED_ADDITIONAL_FIELDS if f not in report]
    if missing_additional:
        problems.append(
            "missing required field(s): %s" % ', '.join(missing_additional)
        )

    if 'evaluation_partition' in report:
        partition = report['evaluation_partition']
        if partition not in VALID_PARTITIONS:
            problems.append(
                "evaluation_partition must be one of %r, got %r"
                % (VALID_PARTITIONS, partition)
            )
        else:
            missing_matrices = [
                partitioned_key(basename, partition)
                for basename in REQUIRED_PARTITION_MATRIX_BASENAMES
                if partitioned_key(basename, partition) not in report
            ]
            if missing_matrices:
                problems.append(
                    "missing required partition-specific field(s) for evaluation_partition=%r: %s"
                    % (partition, ', '.join(missing_matrices))
                )

    if 'global_model' in report:
        global_model = report['global_model']
        if not isinstance(global_model, dict):
            problems.append("global_model must be a dict, got %r" % (type(global_model),))
        else:
            missing_gm_fields = [f for f in REQUIRED_GLOBAL_MODEL_FIELDS if f not in global_model]
            if missing_gm_fields:
                problems.append(
                    "global_model missing required sub-field(s): %s" % ', '.join(missing_gm_fields)
                )
            elif global_model['private_residual_zeroed'] is not True:
                problems.append(
                    "global_model['private_residual_zeroed'] must be literally True, got %r"
                    % (global_model['private_residual_zeroed'],)
                )

    if 'residual_zeroing_log' in report:
        log = report['residual_zeroing_log']
        if not isinstance(log, dict) or not log:
            problems.append(
                "residual_zeroing_log must be a non-empty dict mapping each entity to the "
                "list of *_residual tensor names that were actually zeroed"
            )
        else:
            empty_entries = [k for k, v in log.items() if not v]
            if empty_entries:
                problems.append(
                    "residual_zeroing_log has empty (zero tensors touched) entries for: %s -- "
                    "a residual-zeroed comparison built from these would be silently identical "
                    "to the personalised one" % ', '.join(sorted(empty_entries))
                )

    if 'round_selection' in report:
        round_selection = report['round_selection']
        if not isinstance(round_selection, dict):
            problems.append(
                "round_selection must be a dict, got %r"
                % (type(round_selection),)
            )
        else:
            mode = round_selection.get('mode')
            if mode not in VALID_ROUND_SELECTIONS:
                problems.append(
                    "round_selection['mode'] must be one of %r, got %r"
                    % (VALID_ROUND_SELECTIONS, mode)
                )
            if 'round_index' not in round_selection:
                problems.append("round_selection missing 'round_index'")
            elif not isinstance(round_selection['round_index'], int) or isinstance(
                round_selection['round_index'], bool
            ):
                problems.append(
                    "round_selection['round_index'] must be an int, got %r"
                    % (type(round_selection['round_index']),)
                )

    try:
        non_finite_paths = _find_non_finite_paths(report, '')
    except TypeError as exc:
        problems.append("could not scan report for NaN/Inf: %s" % exc)
        non_finite_paths = []
    if non_finite_paths:
        problems.append(
            "non-finite (NaN/Inf) value(s) found at: %s"
            % ', '.join(non_finite_paths)
        )

    if problems:
        raise ReportValidationError(
            "report failed validation with %d problem(s):\n- %s"
            % (len(problems), '\n- '.join(problems))
        )


def add_deprecated_alias(report: dict, new_key: str, old_key: str) -> None:
    """Mutate report in place, adding old_key as a deprecated alias of new_key.

    If new_key is present and old_key is not already present, sets
    report[old_key] = report[new_key] and records the alias under
    report['_deprecated_aliases'][old_key]. No-op if new_key is absent or
    old_key is already present (never silently overwrite an existing key).
    """
    if new_key in report and old_key not in report:
        report[old_key] = report[new_key]
        report.setdefault('_deprecated_aliases', {})[old_key] = {
            'alias_of': new_key,
            'note': 'deprecated, will be removed in a future release; use the new key',
        }


def should_emit_rounds_to_target(evaluated_round_indices: list) -> bool:
    """A rounds-to-target block is only meaningful across multiple rounds.

    A partition evaluated at a single round (e.g. test evaluated only at
    the final round) must have its rounds-to-target block OMITTED entirely
    -- a block computed from one data point trivially reads as "reached at
    the one round that happened, or never", which looks like a real result
    but is purely an artefact of having only one round evaluated.
    """
    return len(set(evaluated_round_indices)) > 1


if __name__ == '__main__':
    import copy

    def _make_minimal_valid_report():
        report = {
            'schema_version': SCHEMA_VERSION,
            'accounting_version': ACCOUNTING_VERSION,
            'evaluation_partition': 'test',
            'round_selection': {'mode': 'last_round', 'round_index': 149},
            'communication': {
                'per_client': {
                    0: {'uploaded_bytes_total': 8406716 * 4},
                },
                'total_bytes': 56067340800,
            },
            'encoder_coverage': {'FLAIR': 7, 'T1ce': 2, 'T1': 5, 'T2': 4},
            'postproc_policy': 'none',
            'hd95_policy': 'mismatch_empty_penalty',
            'client_modalities': {0: ['FLAIR', 'T1ce', 'T1', 'T2']},
            'provenance': {'git_commit': 'deadbeef', 'run_id': 'splitA_150r'},
            'global_model': {
                'dice': {'mean': 0.6, 'n_cases': 50},
                'per_region': {'WT': {'dice_raw': {'mean': 0.6}}},
                'n_cases': 50,
                'partition': 'global_held_out_test',
                'private_residual_zeroed': True,
            },
            'per_case_stats': {'client_0': {'WT': {'dice_raw': {'mean': 0.5, 'n_cases': 6}}}},
            'fairness': {'worst_client': 3, 'best_client': 0, 'std_across_clients': 0.05},
            'residual_zeroing_log': {'client_0': ['fusion_decoder.adapter.conv_weight_residual']},
        }
        for basename in REQUIRED_PARTITION_MATRIX_BASENAMES:
            report[partitioned_key(basename, 'test')] = {0: [0.5, 0.5, 0.5]}
        return report

    # --- partitioned_key ---
    assert partitioned_key('dice_matrix', 'test') == 'test_dice_matrix'
    assert partitioned_key('dice_matrix', 'validation') == 'validation_dice_matrix'
    try:
        partitioned_key('dice_matrix', 'bogus')
        raise AssertionError("partitioned_key should have raised ValueError")
    except ValueError:
        pass

    # --- is_finite_or_null ---
    assert is_finite_or_null(None) is True
    assert is_finite_or_null(True) is True
    assert is_finite_or_null(False) is True
    assert is_finite_or_null(0) is True
    assert is_finite_or_null(-5) is True
    assert is_finite_or_null('hello') is True
    assert is_finite_or_null(1.5) is True
    assert is_finite_or_null(0.0) is True
    assert is_finite_or_null(float('nan')) is False
    assert is_finite_or_null(float('inf')) is False
    assert is_finite_or_null(float('-inf')) is False
    assert is_finite_or_null({'a': 1, 'b': [1, 2, {'c': None}]}) is True
    assert is_finite_or_null({'a': 1, 'b': [1, float('nan')]}) is False
    assert is_finite_or_null((1, 2, 3)) is True
    assert is_finite_or_null((1, float('inf'))) is False
    try:
        import numpy as np
        is_finite_or_null(np.array([1.0, 2.0]))
        raise AssertionError("is_finite_or_null should have raised TypeError for ndarray")
    except TypeError:
        pass
    except ImportError:
        # numpy not available in this environment: fall back to a plain
        # unexpected type to exercise the same code path.
        try:
            is_finite_or_null(object())
            raise AssertionError("is_finite_or_null should have raised TypeError")
        except TypeError:
            pass

    # --- validate_report: minimal valid report passes ---
    good = _make_minimal_valid_report()
    validate_report(good)  # must not raise

    # --- validate_report: missing required field ---
    missing_field = copy.deepcopy(good)
    del missing_field['hd95_policy']
    try:
        validate_report(missing_field)
        raise AssertionError("validate_report should have raised for missing field")
    except ReportValidationError as exc:
        assert 'hd95_policy' in str(exc)

    # --- validate_report: missing partition-specific matrix -----------------
    missing_matrix = copy.deepcopy(good)
    del missing_matrix[partitioned_key('hd95_postproc_matrix', 'test')]
    try:
        validate_report(missing_matrix)
        raise AssertionError("validate_report should have raised for a missing partition matrix")
    except ReportValidationError as exc:
        assert 'test_hd95_postproc_matrix' in str(exc)

    # A report with every metadata field present but ALL matrices missing
    # must still fail -- metadata completeness alone is not enough.
    metadata_only = copy.deepcopy(good)
    for basename in REQUIRED_PARTITION_MATRIX_BASENAMES:
        del metadata_only[partitioned_key(basename, 'test')]
    try:
        validate_report(metadata_only)
        raise AssertionError("validate_report should have raised when all matrices are missing")
    except ReportValidationError as exc:
        msg = str(exc)
        for basename in REQUIRED_PARTITION_MATRIX_BASENAMES:
            assert partitioned_key(basename, 'test') in msg

    # --- validate_report: missing global_model / per_case_stats / fairness /
    # residual_zeroing_log ----------------------------------------------------
    for field in ('global_model', 'per_case_stats', 'fairness', 'residual_zeroing_log'):
        missing_additional = copy.deepcopy(good)
        del missing_additional[field]
        try:
            validate_report(missing_additional)
            raise AssertionError("validate_report should have raised for missing %r" % field)
        except ReportValidationError as exc:
            assert field in str(exc)

    # --- validate_report: global_model missing a required sub-field ---------
    incomplete_global = copy.deepcopy(good)
    del incomplete_global['global_model']['n_cases']
    try:
        validate_report(incomplete_global)
        raise AssertionError("validate_report should have raised for incomplete global_model")
    except ReportValidationError as exc:
        assert 'global_model' in str(exc) and 'n_cases' in str(exc)

    # --- validate_report: global_model claiming residual NOT zeroed --------
    fake_global = copy.deepcopy(good)
    fake_global['global_model']['private_residual_zeroed'] = False
    try:
        validate_report(fake_global)
        raise AssertionError("validate_report should have raised when private_residual_zeroed is False")
    except ReportValidationError as exc:
        assert 'private_residual_zeroed' in str(exc)

    # --- validate_report: residual_zeroing_log empty / has an empty entry ---
    empty_log = copy.deepcopy(good)
    empty_log['residual_zeroing_log'] = {}
    try:
        validate_report(empty_log)
        raise AssertionError("validate_report should have raised for an empty residual_zeroing_log")
    except ReportValidationError as exc:
        assert 'residual_zeroing_log' in str(exc)

    # The exact failure mode this guards against: zeroing silently matched
    # zero tensors for one client (e.g. a renamed parameter) while every
    # other client's log entry is fine -- must still fail, and must name
    # which entity was empty.
    one_empty_entry = copy.deepcopy(good)
    one_empty_entry['residual_zeroing_log'] = {
        'client_0': ['fusion_decoder.adapter.conv_weight_residual'],
        'client_1': [],
    }
    try:
        validate_report(one_empty_entry)
        raise AssertionError("validate_report should have raised for a client with zero zeroed tensors")
    except ReportValidationError as exc:
        assert 'client_1' in str(exc)

    # --- validate_report: NaN nested deep, with a locatable dotted path ---
    nan_report = copy.deepcopy(good)
    nan_report['communication']['per_client'] = {
        0: {'uploaded_bytes_total': float('nan')}
    }
    try:
        validate_report(nan_report)
        raise AssertionError("validate_report should have raised for NaN")
    except ReportValidationError as exc:
        msg = str(exc)
        assert 'uploaded_bytes_total' in msg
        assert 'communication' in msg and 'per_client' in msg

    # --- validate_report: bad evaluation_partition ---
    bad_partition = copy.deepcopy(good)
    bad_partition['evaluation_partition'] = 'training'
    try:
        validate_report(bad_partition)
        raise AssertionError("validate_report should have raised for bad partition")
    except ReportValidationError as exc:
        assert 'evaluation_partition' in str(exc)

    # --- validate_report: bad round_selection mode / missing round_index ---
    bad_round_mode = copy.deepcopy(good)
    bad_round_mode['round_selection'] = {'mode': 'first_round', 'round_index': 0}
    try:
        validate_report(bad_round_mode)
        raise AssertionError("validate_report should have raised for bad round_selection mode")
    except ReportValidationError as exc:
        assert 'round_selection' in str(exc)

    missing_round_index = copy.deepcopy(good)
    missing_round_index['round_selection'] = {'mode': 'last_round'}
    try:
        validate_report(missing_round_index)
        raise AssertionError("validate_report should have raised for missing round_index")
    except ReportValidationError as exc:
        assert 'round_index' in str(exc)

    non_int_round_index = copy.deepcopy(good)
    non_int_round_index['round_selection'] = {'mode': 'last_round', 'round_index': '149'}
    try:
        validate_report(non_int_round_index)
        raise AssertionError("validate_report should have raised for non-int round_index")
    except ReportValidationError as exc:
        assert 'round_index' in str(exc)

    # --- validate_report: multiple problems reported together ---
    multi_bad = copy.deepcopy(good)
    del multi_bad['hd95_policy']
    del multi_bad['postproc_policy']
    multi_bad['evaluation_partition'] = 'bogus'
    try:
        validate_report(multi_bad)
        raise AssertionError("validate_report should have raised for multi_bad")
    except ReportValidationError as exc:
        msg = str(exc)
        assert 'hd95_policy' in msg
        assert 'postproc_policy' in msg
        assert 'evaluation_partition' in msg

    # --- add_deprecated_alias: exact example from spec ---
    alias_report = {'residual_zeroed_minus_personalised': {0: [0.1, 0.2, 0.3]}}
    add_deprecated_alias(
        alias_report, 'residual_zeroed_minus_personalised', 'test_global_minus_client'
    )
    assert (
        alias_report['test_global_minus_client']
        == alias_report['residual_zeroed_minus_personalised']
    )
    assert (
        alias_report['_deprecated_aliases']['test_global_minus_client']['alias_of']
        == 'residual_zeroed_minus_personalised'
    )

    # add_deprecated_alias: no-op when new_key absent
    no_new_key = {'something_else': 1}
    add_deprecated_alias(no_new_key, 'nonexistent', 'nonexistent_old')
    assert 'nonexistent_old' not in no_new_key
    assert '_deprecated_aliases' not in no_new_key

    # add_deprecated_alias: no-op / no overwrite when old_key already present
    already_has_old = {'new_key': 'new_value', 'old_key': 'preexisting'}
    add_deprecated_alias(already_has_old, 'new_key', 'old_key')
    assert already_has_old['old_key'] == 'preexisting'
    assert '_deprecated_aliases' not in already_has_old

    # --- should_emit_rounds_to_target ---
    assert should_emit_rounds_to_target([149]) is False
    assert should_emit_rounds_to_target([9, 19, 29, 149]) is True
    assert should_emit_rounds_to_target([]) is False
    assert should_emit_rounds_to_target([9, 9, 9]) is False

    # --- REQUIRED_TOP_LEVEL_FIELDS / REQUIRED_ADDITIONAL_FIELDS sanity ---
    for name in (
        'schema_version', 'accounting_version', 'evaluation_partition',
        'round_selection', 'communication', 'encoder_coverage',
        'postproc_policy', 'hd95_policy', 'client_modalities', 'provenance',
    ):
        assert name in REQUIRED_TOP_LEVEL_FIELDS
    for name in ('global_model', 'per_case_stats', 'fairness', 'residual_zeroing_log'):
        assert name in REQUIRED_ADDITIONAL_FIELDS
    for name in ('dice_matrix', 'dice_postproc_matrix', 'hd95_matrix',
                 'hd95_postproc_matrix', 'hd95_valid_pairs_only_matrix',
                 'personalisation_gain_matrix'):
        assert name in REQUIRED_PARTITION_MATRIX_BASENAMES

    print("ALL TESTS PASSED")
