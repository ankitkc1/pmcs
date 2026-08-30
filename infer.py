import os
import csv
import pathlib
import logging
import torch
from torch.utils.data import DataLoader

from options import args_parser
from models.fusion_net import FusionSegNet
from utils.predict import local_test
from dataset.datasets import Brats_test
from train_federated import resolve_modality_masks, build_client_split_files, MODAL_NAMES


def load_checkpoint(model, resume_path, device, client_idx):
    """
    Two checkpoint shapes come out of train_federated.py:
      - a per-client best-model file: {'round', 'dice', 'state_dict'}
      - the periodic full snapshot ('last.pth'): {'clients_dict': [state_dict, ...], ...}
    `client_idx` (1-based) picks the right entry out of the latter.
    """
    ckpt = torch.load(resume_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    elif 'clients_dict' in ckpt:
        model.load_state_dict(ckpt['clients_dict'][client_idx - 1])
    else:
        raise ValueError('unrecognised checkpoint at {!r}: expected a "state_dict" or "clients_dict" key'.format(resume_path))
    return model


if __name__ == '__main__':
    args = args_parser()

    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    masks = resolve_modality_masks(args.setting_options)
    masks_torch = torch.tensor(masks)
    client_idx = args.maskid   # 1-based: which client's model/split to evaluate
    if not (1 <= client_idx < len(masks)):
        raise ValueError('--maskid must select one of the {} per-client mask rows (1..{}); there is no server row anymore.'.format(len(masks) - 1, len(masks) - 1))

    args.save_root = 'test_results'
    args.version = os.path.basename(os.path.dirname(os.path.dirname(args.resume_path)))
    name = os.path.splitext(os.path.basename(args.resume_path))[0]
    args.save_path = os.path.join(args.save_root, args.version)
    os.makedirs(args.save_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=os.path.join(args.save_path, name + '.txt'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    logging.info(masks_torch.int())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FusionSegNet(num_cls=args.num_class).to(args.device)
    load_checkpoint(model, args.resume_path, args.device, client_idx)

    test_file = build_client_split_files(args.setting_options, args.dataname, client_idx)[client_idx]
    test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, modal='all', test_file=test_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    logging.info(str(args))
    logging.info('evaluating client {} (modal mask {})'.format(client_idx, masks[client_idx]))
    logging.info('test dataset size: {}'.format(len(test_set)))

    result_store_path = os.path.join(args.save_path, str(client_idx))
    pathlib.Path(result_store_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.save_path, name + '.csv'), 'w', newline='') as result_file:
        csv_writer = csv.writer(result_file)

        model.eval()
        with torch.no_grad():
            logging.info(' ########## test the model ########## ')
            test_dice_score = local_test(args, test_loader, model, args.device, args.dataname, {}, masks[client_idx], csv_writer, result_store_path)
            test_avg_dice = sum(test_dice_score) / len(test_dice_score)
            logging.info('--- Test Avg_Scores: {:.4f}, cls_Dice: {}'.format(test_avg_dice * 100, test_dice_score))
