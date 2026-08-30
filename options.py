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
    parser.add_argument('--gpus', default='1,2,3,4', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=300, help="number of federated rounds")
    parser.add_argument('--start_round', type=int, default=0, help="round to resume training from")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs per round")
    parser.add_argument('--client_num', type=int, default=4, help="number of federated clients")
    parser.add_argument('--eval', type=int, default=10, help="evaluate every N rounds")
    parser.add_argument('--lam_sd', default=0.1, type=float, help='weight of the modality-dropout self-distillation loss')

    parser.add_argument('--version', type=str, default='debug', help='experiment name, used as the results subfolder')

    # inference
    parser.add_argument('--resume_path', default="", type=str, help='checkpoint to load for inference')
    parser.add_argument('--maskid', default=0, type=int, help='which modality-availability preset row to evaluate')

    args = parser.parse_args()
    return args
