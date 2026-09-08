import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import os
import random
import numpy as np
import time
import collections
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import logging
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn
import copy
import json


from models.fusion_net import FusionSegNet
from utils.fl_utils import avg_EW
from utils import criterions
from dataset.datasets import Brats_test, Brats_train
from dataset.global_split import build_global_test_split
from dataset.split_io import load_materialized_split
from options import args_parser
from utils.predict import local_test
from utils import metrics as fedmetrics
from utils import encoder_coverage as coverage_mod

MODAL_NAMES = ['flair', 't1ce', 't1', 't2']

MODALITY_MASK_PRESETS = {
    'm3': {
        'masks': [[True, True, True, True], [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True]],
    },
    'm2': {
        'masks': [[True, True, True, True], [True, True, False, False], [False, True, False, True], [True, False, False, True], [False, True, True, False], [False, False, True, True], [True, False, True, False]],
    },
    'm1': {
        'masks': [[True, True, True, True], [True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]],
    },
    'c8': {
        'masks': [[True, True, True, True], [True, True, True, False], [True, False, True, True], [True, True, False, False], [False, False, True, True],
                  [False, True, False, False], [False, False, False, True], [True, True, True, True], [True, True, True, True]],
    },
}


def resolve_modality_masks(setting_options):
    for key, preset in MODALITY_MASK_PRESETS.items():
        if key in setting_options:
            return preset['masks']
    raise ValueError('no modality-mask preset matches setting_options={!r}'.format(setting_options))


def build_client_split_files(setting_options, dataname, client_num):

    if 'c8' in setting_options:
        subdir = '20_c8_heter_modalnum' if dataname == 'BRATS2020' else '18_c8_heter_modalnum'
    else:
        subdir = '18_c4_c6'
    return {i: os.path.abspath(os.path.join('split', subdir, 'c{}.csv'.format(i))) for i in range(1, client_num + 1)}


def self_cuda(obj, device):
    if isinstance(obj, list):
        return [self_cuda(l, device) for l in obj]
    elif isinstance(obj, dict):
        return {k:self_cuda(obj[k], device) for k in obj}
    elif torch.is_tensor(obj):
        return obj.to(device)
    return obj


def select_clients(round, state):
    """Sample state['clients_per_round'] of state['client_num'] clients
    uniformly at random, without replacement, using state['participation_rng']
    -- a dedicated random.Random instance seeded once from --seed at the
    start of training, never the global `random` module. This keeps client
    selection reproducible on its own and immune to (and never a source of
    interference with) the data-loader/augmentation randomness that local
    training also draws from the global `random` module for (e.g. the
    modality-dropout self-distillation's random.choice(present_idx)).

    Returned indices are always sorted. This is what guarantees exact
    backward compatibility at clients_per_round == client_num: sampling ALL
    N clients without replacement and sorting the result is list(range(N))
    regardless of the RNG's internal state, so full participation behaves
    identically to the pre-partial-participation code, byte for byte.
    """
    n = state['client_num']
    k = state.get('clients_per_round', n)
    rng = state['participation_rng']
    return sorted(rng.sample(range(n), k))


def local_training(args, device, mask, dataloader, model, client_idx, round, optimizer):
    # set mode to train model

    model.train()
    model = model.to(device)
    start = time.time()
    epoch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[], 'sd':[]}
    optim = optimizer.state_dict()
    optimizer.load_state_dict({k:self_cuda(optim[k], device) for k in optim})

    mask = mask.to(device)             # bool tensor, shape [4]
    n_present = int(mask.sum().item())
    present_idx = mask.nonzero(as_tuple=True)[0].tolist()

    # Round-wise learning-rate decay to reduce late-round
    # oscillation under heterogeneous non-IID clients.
    if round >= 40:
        current_lr = 5e-5
    elif round >= 30:
        current_lr = 1e-4
    else:
        current_lr = args.lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    for iter in range(args.local_ep):
        batch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[], 'sd':[]}

        for batch_idx, data in enumerate(dataloader):

            vol_batch, msk_batch = data[0].to(device), data[1].to(device)
            names = data[-1]
            msk = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)  # [B, 4], already on device
            model.is_training = True
            optimizer.zero_grad(set_to_none=True)

            # encode once
            x1, x2, x3, x4, per_modal = model.encode(vol_batch)

            # teacher decode: all modalities this client actually has
            fuse_pred, prm_preds, _, fused_repr = model.decode(x1, x2, x3, x4, msk)

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(device)
            prm_dice_loss = torch.zeros(1).float().to(device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            per_modal_preds = torch.stack([model.modality_decoder(*feats) for feats in per_modal], dim=0)
            sep_preds = per_modal_preds[mask, ...]

            sep_cross_loss = torch.zeros(1).float().to(device)
            sep_dice_loss = torch.zeros(1).float().to(device)
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss = fuse_loss + prm_loss + sep_loss

            # Normalized cosine modality-dropout self-distillation.
            # Teacher uses all modalities available to the client.
            # Student randomly drops one available modality.
            if n_present > 1:
                drop_idx = random.choice(present_idx)
                student_msk = msk.clone()
                student_msk[:, drop_idx] = False

                _, _, _, fused_repr_student = model.decode(
                    x1, x2, x3, x4, student_msk
                )

                teacher_repr = F.normalize(
                    fused_repr.detach(), p=2, dim=1, eps=1e-6
                )
                student_repr = F.normalize(
                    fused_repr_student, p=2, dim=1, eps=1e-6
                )

                sd_loss = 1.0 - F.cosine_similarity(
                    student_repr, teacher_repr, dim=1
                ).mean()

                sd_warmup = 10
                sd_ramp = 20

                if round < sd_warmup:
                    lambda_sd = 0.0
                else:
                    lambda_sd = args.lam_sd * min(
                        1.0,
                        (round - sd_warmup + 1) / float(sd_ramp)
                    )

                loss = loss + lambda_sd * sd_loss
            else:
                sd_loss = torch.zeros(1).float().to(device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


            batch_loss['total'].append(loss.item())
            batch_loss['fuse'].append(fuse_loss.item())
            batch_loss['prm'].append(prm_loss.item())
            batch_loss['sep'].append(sep_loss.item())
            batch_loss['sd'].append(sd_loss.item())

        for k in epoch_loss:
            epoch_loss[k].append(sum(batch_loss[k])/len(batch_loss[k]))

    for k in epoch_loss:
        epoch_loss[k] = sum(epoch_loss[k])/len(epoch_loss[k])

    msg = 'client_{} local training total time: {:.4f} hours'.format(client_idx+1, (time.time() - start)/3600)
    msg_2 = 'client_{} local training loss: {:.4f} (sd: {:.4f})'.format(client_idx+1, epoch_loss['total'], epoch_loss['sd'])
    print(msg)
    print(msg_2)


    model = model.cpu()

    encoders = [
        model.flair_encoder.state_dict(),
        model.t1ce_encoder.state_dict(),
        model.t1_encoder.state_dict(),
        model.t2_encoder.state_dict()
    ]

    decoder = model.fusion_decoder.state_dict()
    model_state = model.state_dict()

    # Return optimizer state on CPU to avoid CUDA IPC memory retention.
    optimizer_state = optimizer.state_dict()
    for state in optimizer_state["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cpu()


    return encoders, decoder, epoch_loss, model_state, optimizer_state


def aggregate_encoders(local_encoders, active_clients, masks_torch, global_encoders):
    """
    FedAvg each modality-specific encoder over the active clients that hold
    that modality this round. A modality with zero contributors this round
    keeps its previous global weights.
    """
    contributor_counts = []
    for m in range(4):
        client_masks = [masks_torch[c][m] for c in active_clients]
        count = int(sum(bool(cm) for cm in client_masks))
        contributor_counts.append(count)
        if count > 0:
            client_states = [local_encoders[c][m] for c in active_clients]
            global_encoders[m] = avg_EW(client_states, client_masks)
    return global_encoders, contributor_counts


def aggregate_decoder(local_decoders, active_clients):
    """
    Simple FedAvg of the full fusion decoder across all active clients,
    excluding the FusionAdapter's residual parameters, which are trained
    locally and never uploaded.
    """
    keys = local_decoders[active_clients[0]].keys()
    avg_state = collections.OrderedDict()
    for key in keys:
        if key.endswith('_residual'):
            continue
        stacked = torch.stack([local_decoders[c][key].float().cpu() for c in active_clients], dim=0)
        avg_state[key] = stacked.mean(dim=0)
    return avg_state


ENCODER_ATTRS = ['flair_encoder', 't1ce_encoder', 't1_encoder', 't2_encoder']


def broadcast_weights(model_clients, global_encoders, global_decoder_prior, masks, active_clients):
    """Send client k only the global encoders for modalities in its mask,
    plus A0 and the decoder -- never an encoder for a modality it doesn't
    hold (that client never trains it, so shipping it is wasted bandwidth),
    and never the private residual (R_k), which is absent from
    global_decoder_prior and therefore left untouched either way.

    Only clients in `active_clients` are touched at all. A client that sat
    out this round downloads nothing -- its entire local state (encoders,
    decoder, and especially its private residual, which this function never
    writes to for ANY client) is left completely untouched until it is
    selected again in some future round.
    """
    for c in active_clients:
        m = model_clients[c]
        mask = masks[c]
        for held, attr, state in zip(mask, ENCODER_ATTRS, global_encoders):
            if held:
                getattr(m, attr).load_state_dict(state)
        m.fusion_decoder.load_state_dict(global_decoder_prior, strict=False)


def log_round_stats(round, contributor_counts, agg_state, writer=None):
    for m in range(4):
        if contributor_counts[m] > 0:
            agg_state['update_count'][m] += 1
            agg_state['staleness'][m] = 0
        else:
            agg_state['staleness'][m] += 1

        logging.info('Round {} | encoder[{}] contributors={} update_count={} staleness={}'.format(
            round, MODAL_NAMES[m], contributor_counts[m], agg_state['update_count'][m], agg_state['staleness'][m]))

        if writer is not None:
            writer.add_scalar('EncoderAgg/contributors_' + MODAL_NAMES[m], contributor_counts[m], round)
            writer.add_scalar('EncoderAgg/update_count_' + MODAL_NAMES[m], agg_state['update_count'][m], round)
            writer.add_scalar('EncoderAgg/staleness_' + MODAL_NAMES[m], agg_state['staleness'][m], round)


if __name__ == '__main__':
    ### client model = 4 modality-specific encoders + FusionDecoder(prior, aggregated every round) + FusionAdapter(residual, local-only)
    ### encoders are FedAvg'd per modality; the fusion decoder's prior is FedAvg'd across all clients; the adapter residual never leaves the client.
    args = args_parser()

    # data preprocessing follows the RFNet augmentation recipe
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.modelfile_path = os.path.join(args.save_path, 'model_files')
    if not os.path.exists(args.modelfile_path):
        os.makedirs(args.modelfile_path)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/fl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))

    ##### modality availability and per-client train/validation/test files
    materialized_split = bool(args.split_config)
    split_metadata = None
    if materialized_split:
        (masks, args.train_file, args.validation_file, args.test_file,
         args.global_test_file, split_metadata) = load_materialized_split(
            args.split_config, args.client_num, args.data_seed)
        logging.info(
            'Loaded materialized Split %s with data_seed=%d, assignment fingerprint=%s',
            split_metadata['split_id'], split_metadata['data_seed'],
            split_metadata['patient_assignment_fingerprint'])
    else:
        masks = resolve_modality_masks(args.setting_options)
        args.train_file = build_client_split_files(
            args.setting_options, args.dataname, args.client_num)
        # Legacy CSVs use the dataset classes' historical internal 60/40 split.
        args.validation_file = dict(args.train_file)
        args.test_file = dict(args.train_file)
        args.global_test_file = None

    masks_torch = torch.from_numpy(np.array(masks, dtype=np.bool_))
    logging.info(masks_torch.int())

    ########## setting seed for deterministic
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.local_devices = []
    if torch.cuda.is_available():
        args.num_devices = torch.cuda.device_count()
        for i in range(args.client_num):
            args.local_devices.append(torch.device('cuda:{}'.format(i % 4)))  # use 4 gpus
    else:
        # CPU-only fallback (e.g. a smoke test on a dev machine): one
        # "device" (the only CPU), used identically by the same
        # branch/pool dispatch logic below -- this only changes anything
        # when no CUDA device exists at all, so real GPU runs are
        # byte-for-byte unaffected.
        args.num_devices = 1
        for i in range(args.client_num):
            args.local_devices.append(torch.device('cpu'))

    ########## setting model
    client_model = FusionSegNet(num_cls=args.num_class)

    ########## FL setting ##########
    # define dataset, model, optimizer for each clients
    dataloader_clients, validloader_clients, testloader_clients = [], [], []
    model_clients = []
    optimizer_clients = []

    logging.info(str(args))

    for client_idx in range(args.client_num):
        lc_train_file = args.train_file[client_idx+1]
        lc_validation_file = args.validation_file[client_idx+1]
        lc_test_file = args.test_file[client_idx+1]
        data_set = Brats_train(transforms=args.train_transforms, root=args.datapath,
                                modal='all', num_cls=args.num_class, train_file=lc_train_file,
                                all_=materialized_split)
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        valid_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                                modal='all', test_file=lc_validation_file,
                                all_=materialized_split)
        valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                              modal='all', test_file=lc_test_file,
                              all_=materialized_split)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Set Optimizer for the local model update
        net = copy.deepcopy(client_model)
        model_clients.append(net)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_clients.append(optimizer)
        dataloader_clients.append(data_loader)
        validloader_clients.append(valid_loader)
        testloader_clients.append(test_loader)
        logging.info('Client-{} : Brats dataset with modal {}'.format(client_idx+1, masks[client_idx]))
        logging.info('Client-{} cases: train={}, validation={}, test={}'.format(
            client_idx + 1, len(data_set), len(valid_set), len(test_set)))

    best_dices = [0.0] * args.client_num
    best_rounds = [-1] * args.client_num  # 0-indexed round (matches metrics.json round keys)

    ########## additional metrics (opt-in; see options.py) ##########
    voxel_spacing = tuple(args.voxel_spacing)

    private_fraction_static = {}
    for client_idx in range(args.client_num):
        private_fraction_static[client_idx] = fedmetrics.compute_private_fraction(client_model, masks[client_idx])
        pf = private_fraction_static[client_idx]
        logging.info(
            'Client-{} private/shared params: private(R_k)={}, shared(held encoders+A0+decoder)={}, '
            'frac_of_total_model={:.4f}, frac_of_fusion_module={:.4f}'.format(
                client_idx + 1, pf['private_numel'], pf['shared_numel'],
                pf['fraction_of_total_model'], pf['fraction_of_fusion_module']))

    metrics_recorder = fedmetrics.MetricsRecorder(args.save_path, args.client_num)
    metrics_recorder.init_target_tracker(args.target_dice)
    if args.resume != 0:
        metrics_recorder.try_resume()
    metrics_recorder.set_static('client_modalities', {c: masks[c] for c in range(args.client_num)})
    metrics_recorder.set_static('private_param_fraction', private_fraction_static)
    if split_metadata is not None:
        metrics_recorder.set_static('split', split_metadata)
    metrics_recorder.set_static('config', {
        'setting_options': args.setting_options, 'dataname': args.dataname, 'client_num': args.client_num,
        'c_rounds': args.c_rounds, 'eval': args.eval, 'seed': args.seed, 'target_dice': args.target_dice,
        'split_config': os.path.abspath(args.split_config) if args.split_config else '',
        'data_seed': args.data_seed,
        'voxel_spacing': list(voxel_spacing), 'compute_hd95': args.compute_hd95,
        'compute_pers_gain': args.compute_pers_gain, 'eval_global_model': args.eval_global_model,
        'global_test_size': args.global_test_size, 'global_test_seed': args.global_test_seed,
        'postproc': args.postproc, 'min_component_voxels': args.min_component_voxels,
    })

    ##### per-modality-encoder coverage instrumentation (Task 6): tracks,
    ##### every round, whether each modality's encoder got a FedAvg
    ##### contributor, how many, and how many consecutive rounds it has gone
    ##### without one. Pure bookkeeping fed by contributor_counts that
    ##### aggregate_encoders() already computes each round; never touches
    ##### training/aggregation itself.
    encoder_pool_sizes = {
        name: sum(1 for mask in masks if mask[i])
        for i, name in enumerate(coverage_mod.MODALITY_NAMES)
    }
    metrics_recorder.set_static('encoder_pool_sizes', encoder_pool_sizes)
    coverage_tracker = coverage_mod.EncoderCoverageTracker(coverage_mod.MODALITY_NAMES)
    if args.resume != 0:
        # Rebuild from whatever per-round contributor_counts try_resume()
        # already loaded into metrics_recorder.rounds, rather than requiring
        # a separate checkpointed tracker state.
        for r in sorted(metrics_recorder.rounds.keys()):
            counts = metrics_recorder.rounds[r].get('contributor_counts')
            if counts is not None:
                coverage_tracker.record_round(r, counts)

    global_test_loader = None
    if args.eval_global_model:
        if materialized_split:
            global_test_csv = args.global_test_file
        else:
            split_dir = os.path.dirname(next(iter(args.train_file.values())))
            global_test_csv = build_global_test_split(
                datapath=args.datapath, client_split_files=args.train_file, out_dir=split_dir,
                num_cases=args.global_test_size, seed=args.global_test_seed)
        global_test_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                                      modal='all', test_file=global_test_csv, all_=True)
        global_test_loader = DataLoader(dataset=global_test_set, batch_size=1, shuffle=False,
                                         num_workers=0, pin_memory=True)
        logging.info('global held-out test split: {} cases from {}'.format(len(global_test_set), global_test_csv))

    ########## bookkeeping for federated aggregation of the modality encoders
    global_encoders = [client_model.flair_encoder.state_dict(), client_model.t1ce_encoder.state_dict(),
                        client_model.t1_encoder.state_dict(), client_model.t2_encoder.state_dict()]
    global_decoder_prior = collections.OrderedDict(
        (k, v) for k, v in client_model.fusion_decoder.state_dict().items() if not k.endswith('_residual'))
    if args.clients_per_round is None:
        args.clients_per_round = args.client_num
    if not (1 <= args.clients_per_round <= args.client_num):
        raise ValueError('--clients_per_round must be between 1 and --client_num ({}), got {}'.format(
            args.client_num, args.clients_per_round))

    agg_state = {
        'client_num': args.client_num,
        'clients_per_round': args.clients_per_round,
        # Dedicated RNG stream for participation selection only -- never the
        # global `random` module, which local_training's modality-dropout
        # self-distillation already draws from (random.choice(present_idx)).
        # Seeded from --seed so selection is reproducible on its own.
        'participation_rng': random.Random(args.seed),
        'update_count': [0, 0, 0, 0],
        'staleness': [0, 0, 0, 0],
    }

    if args.resume != 0:

        ckpt = torch.load(args.modelfile_path + '/last.pth', map_location='cpu')

        if split_metadata is not None:
            saved_split = ckpt.get('split_metadata')
            if saved_split is None:
                raise ValueError('refusing to resume: checkpoint has no split metadata')
            identity_keys = (
                'split_id', 'data_seed', 'mapping_fingerprint',
                'patient_assignment_fingerprint', 'masks')
            for key in identity_keys:
                if saved_split.get(key) != split_metadata.get(key):
                    raise ValueError(
                        'refusing to resume: checkpoint split mismatch for {}'.format(key))

        for client_i in range(args.client_num):
            model_clients[client_i].load_state_dict(ckpt["clients_dict"][client_i])
            optimizer_clients[client_i].load_state_dict(ckpt["clients_optim_dict"][client_i])

        args.start_round = ckpt['round']
        global_encoders = ckpt['global_encoders']
        global_decoder_prior = ckpt['global_decoder_prior']
        agg_state = ckpt['agg_state']
        # Backward compatibility: a checkpoint saved before partial
        # participation existed has no clients_per_round/participation_rng_state
        # in its agg_state -- backfill full participation so resuming an old
        # run behaves exactly as it always did.
        agg_state.setdefault('clients_per_round', args.client_num)
        rng_state = agg_state.pop('participation_rng_state', None)
        agg_state['participation_rng'] = random.Random(args.seed)
        if rng_state is not None:
            agg_state['participation_rng'].setstate(rng_state)
        best_dices = ckpt['best_dices']
        best_rounds = ckpt.get('best_rounds', [-1] * args.client_num)

        print("load best result: {}".format(best_dices))

    ########## FL Training ##########
    if args.start_round > args.c_rounds:
        raise ValueError('checkpoint round exceeds --c_rounds')

    # c_rounds is a count: a fresh 2-round smoke test executes rounds 0 and 1,
    # then stores completed round 2.  On resume, ckpt['round'] is the next index.
    for round in tqdm(range(args.start_round, args.c_rounds)):
        start = time.time()
        completed_round = round + 1
        is_final_round = completed_round == args.c_rounds
        should_evaluate = (completed_round % args.eval == 0) or is_final_round

        active_clients = select_clients(round, agg_state)
        logging.info('\n | Federated Round : {} | active clients: {} |'.format(round, [c+1 for c in active_clients]))
        # Persisted every round (not just eval rounds), same reasoning as
        # contributor_counts below: flush() writes the whole in-memory
        # self.rounds dict each time, so this reaches disk at the next flush
        # regardless of which round it happened on.
        metrics_recorder.record_round(round, {'selected_clients': active_clients})

        ##### local training (parallel across available GPUs)
        local_encoders = {}
        local_decoders = {}
        result = []
        result_client_ids = []

        num_active = len(active_clients)
        branch_num = num_active // args.num_devices
        if num_active % args.num_devices:
            branch_num += 1

        for branch in range(branch_num):

            ctx = torch.multiprocessing.get_context("spawn")

            pool = ctx.Pool(args.num_devices)

            for slot in range(args.num_devices):
                idx = slot + branch * args.num_devices

                if idx >= num_active:
                        break

                client_i = active_clients[idx]

                result.append(pool.apply_async(local_training, args=(args, args.local_devices[client_i], masks_torch[client_i], dataloader_clients[client_i], model_clients[client_i], client_i, round, optimizer_clients[client_i], )))
                result_client_ids.append(client_i)

            pool.close()
            pool.join()

        logging.info("client training: {}".format(time.time() - start))
        for client_i, r in zip(result_client_ids, result):
            encoders, decoder, loss, model_state, optim = r.get()
            local_encoders[client_i] = encoders
            local_decoders[client_i] = decoder
            model_clients[client_i].load_state_dict(model_state)
            optimizer_clients[client_i].load_state_dict(optim)

            metrics_recorder.comm.record_upload(client_i, round, encoders, decoder, masks[client_i])

            writer.add_scalar('LocalTrain/total_Loss/client_' + str(client_i + 1), loss['total'], round)
            writer.add_scalar('LocalTrain/Loss_fuse/client_' + str(client_i + 1), loss['fuse'], round)
            writer.add_scalar('LocalTrain/Loss_prm/client_' + str(client_i + 1), loss['prm'], round)
            writer.add_scalar('LocalTrain/Loss_sep/client_' + str(client_i + 1), loss['sep'], round)
            writer.add_scalar('LocalTrain/Loss_sd/client_' + str(client_i + 1), loss['sd'], round)

        ##### federated aggregation: modality encoders (per-modality FedAvg) +
        ##### the full fusion decoder's prior (FedAvg across all active clients).
        ##### The FusionAdapter's residual half never leaves the client.
        global_encoders, contributor_counts = aggregate_encoders(local_encoders, active_clients, masks_torch, global_encoders)
        global_decoder_prior = aggregate_decoder(local_decoders, active_clients)

        log_round_stats(round, contributor_counts, agg_state, writer=writer)
        coverage_tracker.record_round(round, contributor_counts)
        # Persisted every round (not just eval rounds) so a resume can replay
        # the tracker exactly; flush() below writes the whole in-memory
        # self.rounds dict each time, so this reaches disk at the next flush
        # regardless of which round it happened on.
        metrics_recorder.record_round(round, {'contributor_counts': contributor_counts})

        broadcast_weights(model_clients, global_encoders, global_decoder_prior, masks, active_clients)

        # Only clients selected this round are counted as having downloaded
        # anything -- at full participation this is every client, but a
        # future reduced-participation sweep must not silently bill an idle
        # client for bytes it never actually received this round.
        for client_i in active_clients:
            metrics_recorder.comm.record_download(
                client_i, round, global_encoders, global_decoder_prior, masks[client_i])

        ##### Eval the model after aggregation and every args.eval rounds
        if should_evaluate:
            logging.info('-'*20 + 'Validate all client models'+ '-'*20)
            with torch.no_grad():
                results = []
                dice_matrix = [None] * args.client_num
                branch_num = args.client_num // args.num_devices
                if args.client_num % args.num_devices:
                    branch_num += 1

                for branch in range(branch_num):

                    ctx = torch.multiprocessing.get_context("spawn")

                    pool = ctx.Pool(args.num_devices)

                    for c_ in range(args.num_devices):
                        c = branch * args.num_devices + c_

                        if c >= args.client_num:
                            break

                        results.append(pool.apply_async(local_test, (args, validloader_clients[c], model_clients[c], args.local_devices[c], 'BRATS2020', {}, masks[c],)))

                    pool.close()
                    pool.join()

                for c, result in enumerate(results):
                    dice_score = result.get()
                    dice_matrix[c] = dice_score
                    c_model = model_clients[c]
                    avgdice_score = sum(dice_score)/len(dice_score)
                    logging.info('--- Validation at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                                        .format((round), avgdice_score*100, dice_score))
                    writer.add_scalar('Eval_AvgDice/client_'+str(c+1), avgdice_score*100, round)

                    if best_dices[c] < avgdice_score:
                        best_dices[c] = avgdice_score
                        best_rounds[c] = round
                        torch.save({
                            'round': round+1,
                            'dice': dice_score,
                            'state_dict': c_model.state_dict(),
                            'split_metadata': split_metadata,
                        }, args.modelfile_path + '/client-%d_round_%d_model_best.pth'%(c+1, round))

                ##### structured metrics.json/.csv -- additive only, never feeds back into
                ##### training/aggregation/best_dices/checkpointing above.
                round_payload = {
                    'dice_matrix': [list(map(float, dice_matrix[c])) for c in range(args.client_num)],
                    'validation_dice_matrix': [list(map(float, dice_matrix[c])) for c in range(args.client_num)],
                    'evaluation_partition': 'validation',
                    'comm_per_client': metrics_recorder.comm.round_totals(round),
                }

                variation = fedmetrics.client_dice_variation(dice_matrix)
                round_payload['client_variation'] = variation
                logging.info(
                    '--- Eval at round_{}, client-Dice variation: std={:.4f} range={:.4f} '
                    'best=client_{} ({:.4f}) worst=client_{} ({:.4f})'.format(
                        round, variation['mean_dice_std'], variation['mean_dice_range'],
                        variation['best_client'] + 1, variation['best_client_mean_dice'],
                        variation['worst_client'] + 1, variation['worst_client_mean_dice']))

                # Client test sets are never used for checkpoint selection or
                # tuning.  They are evaluated once, after the final round, only
                # to produce the reported result.
                test_dice_matrix = None
                if is_final_round:
                    test_results = fedmetrics.run_pooled(
                        args, list(range(args.client_num)), local_test,
                        lambda c: (args, testloader_clients[c], model_clients[c],
                                   args.local_devices[c], 'BRATS2020', {}, masks[c]))
                    test_dice_matrix = [test_results[c] for c in range(args.client_num)]
                    round_payload['test_dice_matrix'] = [
                        list(map(float, test_dice_matrix[c])) for c in range(args.client_num)]
                    round_payload['test_client_variation'] = fedmetrics.client_dice_variation(
                        test_dice_matrix)
                    for c in range(args.client_num):
                        logging.info(
                            '--- FINAL TEST at round_{}, client_{}, Avg_Scores: {:.4f}, cls_Dice: {}'.format(
                                round, c + 1, float(np.mean(test_dice_matrix[c])) * 100.0,
                                test_dice_matrix[c]))

                # HD95 is a reporting metric, not a tuning signal.  Compute it
                # only once at the final round; validation Dice remains periodic.
                # Every Dice/HD95 number here is reported twice -- raw and after
                # --postproc -- so the effect of post-processing can be read off
                # directly instead of guessed at.
                if args.compute_hd95 and is_final_round:
                    hd95_results = fedmetrics.run_pooled(
                        args, list(range(args.client_num)), fedmetrics.evaluate_client,
                        lambda c: (args, validloader_clients[c], model_clients[c], args.local_devices[c],
                                   masks[c], True, voxel_spacing, args.postproc, args.min_component_voxels))
                    hd95_matrix = [hd95_results[c]['hd95'] for c in range(args.client_num)]
                    round_payload['hd95_matrix'] = [list(map(float, v)) for v in hd95_matrix]
                    round_payload['hd95_edge_counts'] = {c: hd95_results[c]['edge_counts'] for c in range(args.client_num)}
                    round_payload['hd95_valid_pairs_only_matrix'] = [
                        hd95_results[c]['hd95_valid_pairs_only']
                        for c in range(args.client_num)]
                    round_payload['hd95_policy'] = hd95_results[0]['hd95_policy']
                    round_payload['dice_postproc_matrix'] = [
                        list(map(float, hd95_results[c]['dice_postproc'])) for c in range(args.client_num)]
                    round_payload['hd95_postproc_matrix'] = [
                        list(map(float, hd95_results[c]['hd95_postproc'])) for c in range(args.client_num)]
                    round_payload['hd95_postproc_valid_pairs_only_matrix'] = [
                        hd95_results[c]['hd95_postproc_valid_pairs_only']
                        for c in range(args.client_num)]
                    round_payload['postproc_edge_counts'] = {
                        c: hd95_results[c]['postproc_edge_counts'] for c in range(args.client_num)}
                    round_payload['postproc_policy'] = hd95_results[0]['postproc_policy']
                    for c in range(args.client_num):
                        logging.info('--- Eval at round_{}, client_{} HD95(mm): {} | edge cases: {} | '
                                     'postproc({}): Dice={} HD95(mm)={}'.format(
                            round, c + 1, hd95_results[c]['hd95'], hd95_results[c]['edge_counts'],
                            args.postproc, hd95_results[c]['dice_postproc'], hd95_results[c]['hd95_postproc']))

                    if is_final_round:
                        test_hd95_results = fedmetrics.run_pooled(
                            args, list(range(args.client_num)), fedmetrics.evaluate_client,
                            lambda c: (args, testloader_clients[c], model_clients[c],
                                       args.local_devices[c], masks[c], True, voxel_spacing,
                                       args.postproc, args.min_component_voxels))
                        round_payload['test_hd95_matrix'] = [
                            list(map(float, test_hd95_results[c]['hd95']))
                            for c in range(args.client_num)]
                        round_payload['test_hd95_edge_counts'] = {
                            c: test_hd95_results[c]['edge_counts']
                            for c in range(args.client_num)}
                        round_payload['test_hd95_valid_pairs_only_matrix'] = [
                            test_hd95_results[c]['hd95_valid_pairs_only']
                            for c in range(args.client_num)]
                        round_payload['test_dice_postproc_matrix'] = [
                            list(map(float, test_hd95_results[c]['dice_postproc']))
                            for c in range(args.client_num)]
                        round_payload['test_hd95_postproc_matrix'] = [
                            list(map(float, test_hd95_results[c]['hd95_postproc']))
                            for c in range(args.client_num)]
                        round_payload['test_hd95_postproc_valid_pairs_only_matrix'] = [
                            test_hd95_results[c]['hd95_postproc_valid_pairs_only']
                            for c in range(args.client_num)]
                        round_payload['test_postproc_edge_counts'] = {
                            c: test_hd95_results[c]['postproc_edge_counts']
                            for c in range(args.client_num)}

                if args.compute_pers_gain:
                    zero_models = {c: fedmetrics.zero_residual_copy(model_clients[c]) for c in range(args.client_num)}
                    pers_results = fedmetrics.run_pooled(
                        args, list(range(args.client_num)), fedmetrics.evaluate_client,
                        lambda c: (args, validloader_clients[c], zero_models[c], args.local_devices[c],
                                   masks[c], False, voxel_spacing))
                    gain_matrix = [dice_matrix[c] - pers_results[c]['dice'] for c in range(args.client_num)]
                    round_payload['personalisation_gain_matrix'] = [list(map(float, v)) for v in gain_matrix]
                    for c in range(args.client_num):
                        logging.info('--- Eval at round_{}, client_{} personalisation gain (with R_k - without R_k): {}'.format(
                            round, c + 1, gain_matrix[c]))

                    if is_final_round:
                        test_zero_models = {
                            c: fedmetrics.zero_residual_copy(model_clients[c])
                            for c in range(args.client_num)}
                        test_pers_results = fedmetrics.run_pooled(
                            args, list(range(args.client_num)), fedmetrics.evaluate_client,
                            lambda c: (args, testloader_clients[c], test_zero_models[c],
                                       args.local_devices[c], masks[c], False, voxel_spacing))
                        test_gain_matrix = [
                            test_dice_matrix[c] - test_pers_results[c]['dice']
                            for c in range(args.client_num)]
                        round_payload['test_personalisation_gain_matrix'] = [
                            list(map(float, value)) for value in test_gain_matrix]

                if args.eval_global_model:
                    global_model = FusionSegNet(num_cls=args.num_class)
                    global_model.flair_encoder.load_state_dict(global_encoders[0])
                    global_model.t1ce_encoder.load_state_dict(global_encoders[1])
                    global_model.t1_encoder.load_state_dict(global_encoders[2])
                    global_model.t2_encoder.load_state_dict(global_encoders[3])
                    global_model.fusion_decoder.load_state_dict(global_decoder_prior, strict=False)
                    with torch.no_grad():
                        for name, parameter in global_model.named_parameters():
                            if name.endswith('_residual'):
                                parameter.zero_()

                    # client_ceiling[k]: the full global model (all 4 global
                    # encoders + A0, R=0, shared decoder) scored on client k's
                    # OWN validation/test cases using ALL FOUR modalities --
                    # every case has all four in the data even though client k
                    # never trained on the ones outside its own mask. This is
                    # the ceiling a non-personalised, full-modality model
                    # reaches on that client's patients.
                    # modality_deficit[k] = client_ceiling[k] - personalised_score[k]
                    # is the per-client cost of missing modalities (+ lack of
                    # personalisation) on that client's own patients.
                    #
                    # dispatch the per-client pooled eval first, while global_model is
                    # still CPU-resident -- each spawned worker pickles its own copy,
                    # so this never mutates the parent's global_model in place.
                    full_mask = [True, True, True, True]
                    client_ceiling_results = fedmetrics.run_pooled(
                        args, list(range(args.client_num)), fedmetrics.evaluate_client,
                        lambda c: (args, validloader_clients[c], global_model, args.local_devices[c],
                                   full_mask, False, voxel_spacing))
                    client_ceiling = {c: client_ceiling_results[c]['dice'] for c in range(args.client_num)}
                    modality_deficit = {
                        c: (client_ceiling[c] - dice_matrix[c]) for c in range(args.client_num)
                    }
                    round_payload['client_ceiling'] = {
                        c: list(map(float, client_ceiling[c])) for c in range(args.client_num)}
                    round_payload['modality_deficit'] = {
                        c: list(map(float, modality_deficit[c])) for c in range(args.client_num)}
                    round_payload['client_ceiling_partition'] = 'validation'
                    for c in range(args.client_num):
                        logging.info(
                            '--- Eval at round_{}, client_{} client_ceiling (global model, all 4 modalities): {} | '
                            'modality_deficit (ceiling - personalised): {}'.format(
                                round, c + 1, client_ceiling[c], modality_deficit[c]))

                    if is_final_round:
                        test_client_ceiling_results = fedmetrics.run_pooled(
                            args, list(range(args.client_num)), fedmetrics.evaluate_client,
                            lambda c: (args, testloader_clients[c], global_model,
                                       args.local_devices[c], full_mask, False, voxel_spacing))
                        test_client_ceiling = {
                            c: test_client_ceiling_results[c]['dice'] for c in range(args.client_num)}
                        test_modality_deficit = {
                            c: (test_client_ceiling[c] - test_dice_matrix[c])
                            for c in range(args.client_num)}
                        round_payload['test_client_ceiling'] = {
                            c: list(map(float, test_client_ceiling[c])) for c in range(args.client_num)}
                        round_payload['test_modality_deficit'] = {
                            c: list(map(float, test_modality_deficit[c]))
                            for c in range(args.client_num)}

                        # This is the only access to the shared held-out test set.
                        # It runs in-process after all pooled evaluations are done.
                        global_test_result = fedmetrics.evaluate_client(
                            args, global_test_loader, global_model, args.device,
                            full_mask, True, voxel_spacing, args.postproc, args.min_component_voxels)
                        round_payload['global_model'] = {
                            'dice': list(map(float, global_test_result['dice'])),
                            'hd95': list(map(float, global_test_result['hd95'])),
                            'hd95_valid_pairs_only': global_test_result['hd95_valid_pairs_only'],
                            'hd95_policy': global_test_result['hd95_policy'],
                            'edge_counts': global_test_result['edge_counts'],
                            'dice_postproc': list(map(float, global_test_result['dice_postproc'])),
                            'hd95_postproc': list(map(float, global_test_result['hd95_postproc'])),
                            'hd95_postproc_valid_pairs_only': global_test_result['hd95_postproc_valid_pairs_only'],
                            'postproc_policy': global_test_result['postproc_policy'],
                            'postproc_edge_counts': global_test_result['postproc_edge_counts'],
                            'n_cases': global_test_result['n_cases'],
                            'partition': 'global_held_out_test',
                            'private_residual_zeroed': True,
                        }
                        logging.info(
                            '--- FINAL GLOBAL TEST at round_{}, Dice={} HD95(mm)={} | postproc({}): '
                            'Dice={} HD95(mm)={}'.format(
                                round, global_test_result['dice'], global_test_result['hd95'],
                                args.postproc, global_test_result['dice_postproc'],
                                global_test_result['hd95_postproc']))

                if is_final_round:
                    convergence = {}
                    for c in range(args.client_num):
                        not_converged = best_rounds[c] == round
                        convergence[c] = {
                            'best_round': best_rounds[c],
                            'best_validation_dice': best_dices[c],
                            'converged': not not_converged,
                        }
                        if not_converged:
                            logging.warning(
                                'Client_{} had not converged: its best validation round ({}) is the last '
                                'evaluated round ({}); validation Dice may still have been rising when '
                                'training stopped.'.format(c + 1, best_rounds[c] + 1, round + 1))
                    round_payload['convergence'] = convergence

                # Recomputed fresh from the tracker's full accumulated
                # history on every flush (cheap; pure arithmetic over
                # already-recorded per-round contributor counts), so this
                # field is always consistent with metrics_recorder.rounds
                # rather than a stale snapshot from whenever it was last set.
                metrics_recorder.set_static('encoder_coverage', coverage_mod.summarize(
                    coverage_tracker, encoder_pool_sizes, total_clients=args.client_num,
                    participation_k=len(active_clients)))

                # Same "recompute fresh from metrics_recorder.rounds every
                # flush" reasoning as encoder_coverage above.
                selected_by_round = {
                    r: payload['selected_clients']
                    for r, payload in metrics_recorder.rounds.items()
                    if 'selected_clients' in payload
                }
                rounds_participated = {
                    c: sum(1 for selected in selected_by_round.values() if c in selected)
                    for c in range(args.client_num)
                }
                metrics_recorder.set_static('participation', {
                    'K': args.clients_per_round,
                    'N': args.client_num,
                    'selected_by_round': selected_by_round,
                    'rounds_participated': rounds_participated,
                })

                metrics_recorder.record_round(round, round_payload)
                metrics_recorder.flush()

        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        if should_evaluate:
            # random.Random is not on torch.load's weights_only-safe allowlist
            # (PyTorch >=2.6 defaults weights_only=True, and this codebase's
            # own resume call above passes no override) -- save only the
            # plain getstate() tuple, never the live RNG object, so resume
            # keeps working regardless of which torch version this actually
            # runs under.
            saved_agg_state = dict(agg_state)
            saved_agg_state['participation_rng_state'] = agg_state['participation_rng'].getstate()
            del saved_agg_state['participation_rng']

            torch.save({

            'round': round + 1,

            "clients_dict": [model_clients[client_i].state_dict() for client_i in range(args.client_num)],
            "clients_optim_dict": [optimizer_clients[client_i].state_dict() for client_i in range(args.client_num)],

            'global_encoders': global_encoders,
            'global_decoder_prior': global_decoder_prior,
            'agg_state': saved_agg_state,

            'best_dices': best_dices,
            'best_rounds': best_rounds,
            'split_metadata': split_metadata,
            }, args.modelfile_path + '/last.pth')

    writer.close()
