"""Shared loader for materialized paired Split A/B configs.

Used by both train_federated.py (live training) and evaluate.py (offline
re-scoring) so the two can never disagree about which patient goes to which
client, which modalities a client holds, or where the held-out global test
set lives. Moving this out of train_federated.py also lets evaluate.py load
a split without importing the training entrypoint at all.
"""
import hashlib
import json
import os


def _read_case_ids(path):
    with open(path, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


def load_materialized_split(config_path, client_num, data_seed):
    """Load and defensively validate a materialized paired-split config."""
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as handle:
        config = json.load(handle)

    if config.get('modality_order') != ['FLAIR', 'T1ce', 'T1', 'T2']:
        raise ValueError('split config modality order must be [FLAIR, T1ce, T1, T2]')
    if int(config.get('data_seed', -1)) != int(data_seed):
        raise ValueError('split config data_seed does not match --data_seed')
    if client_num != 8 or config.get('split_id') not in ('A', 'B'):
        raise ValueError('paired Split A/B requires exactly 8 clients')

    client_ids = ['C{}'.format(i) for i in range(1, client_num + 1)]
    if sorted(config.get('clients', {})) != sorted(client_ids):
        raise ValueError('split config client set does not match --client_num')
    if sorted(config.get('masks', {})) != sorted(client_ids):
        raise ValueError('split config mask set does not match --client_num')

    masks = []
    train_files, validation_files, test_files = {}, {}, {}
    all_client_cases = set()
    assignment_hasher = hashlib.sha256()
    config_dir = os.path.dirname(config_path)

    def resolve_path(value):
        path = value if os.path.isabs(value) else os.path.join(config_dir, value)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ValueError('split file does not exist: {}'.format(path))
        return path

    for index, client_id in enumerate(client_ids, start=1):
        mask = [bool(value) for value in config['masks'][client_id]]
        if len(mask) != 4 or not any(mask):
            raise ValueError('{} has an invalid/empty modality mask'.format(client_id))
        masks.append(mask)

        record = config['clients'][client_id]
        client_cases = set()
        for partition, destination in (
                ('train', train_files), ('validation', validation_files), ('test', test_files)):
            path = resolve_path(record['files'][partition])
            case_ids = _read_case_ids(path)
            expected = int(record['partition_counts'][partition])
            if len(case_ids) != expected or len(case_ids) != len(set(case_ids)):
                raise ValueError('{} {} count/uniqueness check failed'.format(client_id, partition))
            overlap = client_cases.intersection(case_ids)
            if overlap:
                raise ValueError('{} partitions overlap: {}'.format(client_id, sorted(overlap)))
            client_cases.update(case_ids)
            destination[index] = path
            assignment_hasher.update(
                ('{}|{}|{}\n'.format(client_id, partition, '\n'.join(case_ids))).encode('utf-8'))

        overlap = all_client_cases.intersection(client_cases)
        if overlap:
            raise ValueError('patients occur in multiple clients: {}'.format(sorted(overlap)))
        all_client_cases.update(client_cases)

    expected_split_a_masks = [
        [True, True, True, True], [True, True, True, False],
        [True, False, True, True], [True, False, True, False],
        [True, False, False, True], [True, False, False, False],
        [True, False, True, False], [False, False, False, True],
    ]
    expected_masks = expected_split_a_masks
    if config['split_id'] == 'B':
        expected_masks = [
            [mask[1], mask[0], mask[2], mask[3]]
            for mask in expected_split_a_masks]
    if masks != expected_masks:
        raise ValueError('Split {} masks do not match the registered design'.format(
            config['split_id']))

    actual_pools = {
        name: sum(int(mask[m]) for mask in masks)
        for m, name in enumerate(['FLAIR', 'T1ce', 'T1', 'T2'])
    }
    expected_pools = {name: int(value) for name, value in config['expected_modality_pool_sizes'].items()}
    if actual_pools != expected_pools:
        raise ValueError('modality pool sizes do not match materialized config')

    global_test_file = resolve_path(config['global_test_file'])
    global_cases = _read_case_ids(global_test_file)
    if len(global_cases) != 50 or len(global_cases) != len(set(global_cases)):
        raise ValueError('global held-out test must contain 50 unique cases')
    overlap = all_client_cases.intersection(global_cases)
    if overlap:
        raise ValueError('global held-out test overlaps clients: {}'.format(sorted(overlap)))

    metadata = {
        'split_id': config['split_id'],
        'data_seed': int(config['data_seed']),
        'mapping_fingerprint': config['mapping_fingerprint'],
        'patient_assignment_fingerprint': assignment_hasher.hexdigest(),
        'modality_order': config['modality_order'],
        'masks': masks,
        'global_test_file': global_test_file,
        'expected_modality_pool_sizes': expected_pools,
    }
    return masks, train_files, validation_files, test_files, global_test_file, metadata
