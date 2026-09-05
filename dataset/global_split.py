import glob
import os
import random


def _case_id_from_volpath(path):
    base = os.path.basename(path)
    if base.endswith('_vol.npy'):
        return base[:-len('_vol.npy')]
    return os.path.splitext(base)[0]


def build_global_test_split(datapath, client_split_files, out_dir, num_cases=50, seed=42):
    """
    Deterministically carve a held-out global test split from the unassigned cases in the dataset, and write it to a CSV file.s
    """
    out_path = os.path.join(out_dir, 'global_test.csv')
    if os.path.exists(out_path):
        return out_path

    all_cases = sorted({
        _case_id_from_volpath(p) for p in glob.glob(os.path.join(datapath, 'vol', '*.npy'))
    })

    used_cases = set()
    for f in client_split_files.values():
        with open(f) as fh:
            used_cases.update(line.strip() for line in fh if line.strip())

    free_cases = sorted(c for c in all_cases if c not in used_cases)
    if len(free_cases) < num_cases:
        raise ValueError(
            'requested a {}-case global test split but only {} unassigned cases '
            'exist under {!r} (found {} total cases, {} already assigned to clients)'.format(
                num_cases, len(free_cases), datapath, len(all_cases), len(used_cases)))

    rng = random.Random(seed)
    rng.shuffle(free_cases)
    chosen = sorted(free_cases[:num_cases])

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(chosen) + '\n')
    return out_path
