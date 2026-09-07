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
      (d) no NaN/+-inf anywhere in the tree (dotted-path locations of every
          offender are collected into the error message).
    """
    problems = []

    missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in report]
    if missing:
        problems.append(
            "missing required top-level field(s): %s" % ', '.join(missing)
        )

    if 'evaluation_partition' in report:
        partition = report['evaluation_partition']
        if partition not in VALID_PARTITIONS:
            problems.append(
                "evaluation_partition must be one of %r, got %r"
                % (VALID_PARTITIONS, partition)
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
        return {
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
        }

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

    # --- REQUIRED_TOP_LEVEL_FIELDS sanity ---
    for name in (
        'schema_version', 'accounting_version', 'evaluation_partition',
        'round_selection', 'communication', 'encoder_coverage',
        'postproc_policy', 'hd95_policy', 'client_modalities', 'provenance',
    ):
        assert name in REQUIRED_TOP_LEVEL_FIELDS

    print("ALL TESTS PASSED")
