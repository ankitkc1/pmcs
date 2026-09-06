import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', default='./data/BRATS2018_Training_none_npy', type=str)
    parser.add_argument('--dataname', default='BRATS2018', type=str)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--save_root', default='./results', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')

    # Federated learning settings
    parser.add_argument('--setting_options', default="c8", type=str, help="modality-availability preset: m1 / m2 / m3 / c8")
    parser.add_argument('--split_config', default="", type=str,
                        help='materialized Split A/B config; overrides setting_options and legacy client CSVs')
    parser.add_argument('--data_seed', default=20260905, type=int,
                        help='patient-assignment seed, deliberately independent of --seed')
    parser.add_argument('--gpus', default='1,2,3,4', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=250, help="number of federated rounds")
    parser.add_argument('--start_round', type=int, default=0, help="round to resume training from")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs per round")
    parser.add_argument('--client_num', type=int, default=4, help="number of federated clients")
    parser.add_argument('--eval', type=int, default=10, help="evaluate every N rounds")
    parser.add_argument('--lam_sd', default=0.1, type=float, help='weight of the modality-dropout self-distillation loss')

    parser.add_argument('--version', type=str, default='debug', help='experiment name, used as the results subfolder')

    # inference
    parser.add_argument('--resume_path', default="", type=str, help='checkpoint to load for inference')
    parser.add_argument('--maskid', default=0, type=int, help='which modality-availability preset row to evaluate')

    # additional metrics (all opt-in; none of these change training/aggregation/data loading)
    parser.add_argument('--compute_hd95', action='store_true',
                         help='additionally compute 95th-percentile Hausdorff distance (mm) per region/client/round; requires medpy')
    parser.add_argument('--compute_pers_gain', action='store_true',
                         help="additionally evaluate every client with its private residual R_k zeroed (A0-only), to measure "
                              "personalisation gain = dice_with_Rk - dice_without_Rk; roughly doubles eval cost")
    parser.add_argument('--eval_global_model', action='store_true',
                         help='additionally build a global model (global encoders + A0, R=0, shared decoder) and evaluate it '
                              "on client validation splits, then on client/global held-out test splits only at the final round")
    parser.add_argument('--target_dice', nargs='+', type=float, default=[50, 55, 60, 65],
                         help='mean-Dice(%%) targets to report the first eval round each one is reached at')
    parser.add_argument('--global_test_size', type=int, default=50,
                         help='legacy split only: number of held-out global-test cases')
    parser.add_argument('--global_test_seed', type=int, default=42,
                         help='seed for the deterministic global test split selection')
    parser.add_argument('--voxel_spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                         help='voxel spacing in mm (H, W, D) of the preprocessed volumes, used for HD95')
    parser.add_argument('--postproc', default='largest_cc', choices=['none', 'largest_cc', 'min_volume'],
                         help='post-processing applied to each predicted binary region mask (WT/TC/ET) before '
                              'Dice/HD95, in addition to (never replacing) the raw un-post-processed numbers; '
                              'only takes effect where --compute_hd95 is also set')
    parser.add_argument('--min_component_voxels', type=int, default=50,
                         help='--postproc min_volume: drop connected components smaller than this many voxels')

    args = parser.parse_args()
    return args
