import torch
import os
import random
import numpy as np
import time
import collections
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn
import copy


from models import model
from utils.fl_utils import avg_EW
from utils.lr_scheduler import LR_Scheduler
from utils import criterions
from dataset.datasets import Brats_test, Brats_train
from options import args_parser
from utils.predict import local_test

MODAL_NAMES = ['flair', 't1ce', 't1', 't2']


def self_cuda(obj, device):
    if isinstance(obj, list):
        return [self_cuda(l, device) for l in obj]
    elif isinstance(obj, dict):
        return {k:self_cuda(obj[k], device) for k in obj}
    elif torch.is_tensor(obj):
        return obj.to(device)
    return obj


def select_clients(round, state):
    """
    Decide which clients participate in a given federated round.

    `state` carries whatever bookkeeping a participation policy needs (e.g.
    per-client last-participation round, for async or partial-participation
    schemes). The default policy is full participation: every client trains
    every round.
    """
    return list(range(state['client_num']))


def local_training(args, device, mask, dataloader, model, client_idx, round, optimizer):
    # set mode to train model

    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    model.train()
    model = model.to(device)
    start = time.time()
    epoch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[], 'sd':[]}
    optim = optimizer.state_dict()
    optimizer.load_state_dict({k:self_cuda(optim[k], device) for k in optim})

    step_lr = lr_schedule(optimizer, round)

    mask = mask.to(device)             # bool tensor, shape [4]
    n_present = int(mask.sum().item())
    present_idx = mask.nonzero(as_tuple=True)[0].tolist()

    for iter in range(args.local_ep):
        batch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[], 'sd':[]}

        for batch_idx, data in enumerate(dataloader):

            vol_batch, msk_batch = data[0].to(device), data[1].to(device)
            names = data[-1]
            msk = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)  # [B, 4], already on device
            model.is_training = True

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

            flair_pred = model.decoder_sep(*per_modal[0])
            t1ce_pred = model.decoder_sep(*per_modal[1])
            t1_pred = model.decoder_sep(*per_modal[2])
            t2_pred = model.decoder_sep(*per_modal[3])
            sep_preds = torch.stack((flair_pred, t1ce_pred, t1_pred, t2_pred), dim=0)[mask, ...]

            sep_cross_loss = torch.zeros(1).float().to(device)
            sep_dice_loss = torch.zeros(1).float().to(device)
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss = fuse_loss + prm_loss + sep_loss

            # modality-dropout self-distillation: student decode with one
            # present modality randomly dropped, distilled from the teacher's
            # fused representation. Skipped for single-modality clients.
            if n_present > 1:
                drop_idx = random.choice(present_idx)
                student_msk = msk.clone()
                student_msk[:, drop_idx] = False
                _, _, _, fused_repr_student = model.decode(x1, x2, x3, x4, student_msk)
                sd_loss = F.mse_loss(fused_repr_student, fused_repr.detach())
                loss = loss + args.lam_sd * sd_loss
            else:
                sd_loss = torch.zeros(1).float().to(device)

            optimizer.zero_grad()
            loss.backward()
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
    encoders = [model.c1_encoder.state_dict(), model.c2_encoder.state_dict(),
                model.c3_encoder.state_dict(), model.c4_encoder.state_dict()]
    decoder = model.decoder_fuse.state_dict()
    return encoders, decoder, epoch_loss, model, optimizer.state_dict()


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


def broadcast_weights(model_clients, global_encoders, global_decoder_prior):
    for m in model_clients:
        m.c1_encoder.load_state_dict(global_encoders[0])
        m.c2_encoder.load_state_dict(global_encoders[1])
        m.c3_encoder.load_state_dict(global_encoders[2])
        m.c4_encoder.load_state_dict(global_encoders[3])
        # residual FusionAdapter params are absent from global_decoder_prior
        # and therefore left untouched (never downloaded either).
        m.decoder_fuse.load_state_dict(global_decoder_prior, strict=False)


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
    ### local model - 模态特异Encoder & 模态融合Decoder(prior，全局聚合) + FusionAdapter(residual，本地私有)
    ### FL过程中：Encoder按模态聚合，Decoder的prior部分全量聚合，residual部分从不上传
    args = args_parser()

    # 数据预处理遵循RFNet
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

    ##### modality missing mask
    if "m3" in args.setting_options:
        masks = [[True, True, True, True], [True, True, True,False], [True, False, True, True], [True, True, False, True], [False, True, True, True]]
        mask_name = ['flairt1cet1t2', 'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1t2']
    elif "m2" == args.setting_options:
        masks = [[True, True, True, True], [True, True, False,False],  [False, True, False, True], [True, False, False, True], [False, True, True, False], [False, False, True, True], [True, False, True, False]]
        mask_name = ['flairt1cet1t2', 'flairt1ce', 't1t2', 'flairt1', 't1cet2', 'flairt2', 't1cet1']
    elif "m1" in args.setting_options:
        masks = [[True, True, True, True], [True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
        mask_name = ['flairt1cet1t2', 'flair', 't1ce', 't1', 't2']
    elif "c8" in args.setting_options:
        masks = [[True, True, True, True], [True,  True, True,  False], [True,  False, True,  True], [True, True, False, False], [False, False, True, True],
             [False, True, False, False], [False, False, False, True], [True,  True,  True, True], [True,  True,  True, True]]
        mask_name = ['m1111', 'm1110', 'm1011', 'm1100', 'm0011', 'm0100', 'm0001', 'm1111', 'm1111']

    if "c8" in args.setting_options:
        if args.dataname == "BRATS2020":
            args.train_file = {1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c1.csv",
                                    2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c2.csv",
                                    3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c3.csv",
                                    4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c4.csv",
                                    5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c5.csv",
                                    6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c6.csv",
                                    7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c7.csv",
                                    8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c8.csv"}
        else:
            args.train_file = {1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c1.csv",
                                    2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c2.csv",
                                    3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c3.csv",
                                    4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c4.csv",
                                    5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c5.csv",
                                    6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c6.csv",
                                    7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c7.csv",
                                    8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c8.csv"}
    else:
        args.train_file = {1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c1.csv",
                2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c2.csv",
                3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c3.csv",
                4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c4.csv",
                5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c5.csv",
                6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c6.csv"
                }

        args.valid_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/val.csv"
        args.test_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/test.csv"

    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['flair', 't1ce', 't1', 't2']
    logging.info(masks_torch.int())

    ########## setting seed for deterministic
    if args.deterministic:
        # cudnn.enabled = False
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.local_devices = []
    args.num_devices = torch.cuda.device_count()
    for i in range(args.client_num):
        args.local_devices.append(torch.device('cuda:{}'.format(i%4))) # use 4 gpus

    ########## setting model
    client_model = model.E4D4Model(num_cls=args.num_class)

    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    ########## FL setting ##########
    # define dataset, model, optimizer for each clients
    dataloader_clients, validloader_clients, testloader_clients = [], [], []
    model_clients = []
    optimizer_clients = []

    modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))

    for client_idx in range(args.client_num):
        chose_modal = 'all'
        lc_train_file = args.train_file[client_idx+1]
        data_set = Brats_train(transforms=args.train_transforms, root=args.datapath,
                                modal=chose_modal, num_cls=args.num_class, train_file=lc_train_file)
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        valid_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                                modal=chose_modal, test_file=lc_train_file)
        valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = valid_loader

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
        logging.info('the length of Brats dataset is {} : {}'.format(len(data_set), len(valid_set)))

        device = args.local_devices[client_idx]

    best_dices = [0.0] * args.client_num

    ########## bookkeeping for federated aggregation of the modality encoders
    global_encoders = [client_model.c1_encoder.state_dict(), client_model.c2_encoder.state_dict(),
                        client_model.c3_encoder.state_dict(), client_model.c4_encoder.state_dict()]
    global_decoder_prior = collections.OrderedDict(
        (k, v) for k, v in client_model.decoder_fuse.state_dict().items() if not k.endswith('_residual'))
    agg_state = {
        'client_num': args.client_num,
        'update_count': [0, 0, 0, 0],
        'staleness': [0, 0, 0, 0],
    }

    if args.resume != 0:

        ckpt = torch.load(args.modelfile_path + '/last.pth')

        for client_i in range(args.client_num):
            model_clients[client_i].load_state_dict(ckpt["clients_dict"][client_i])
            optimizer_clients[client_i].load_state_dict(ckpt["clients_optim_dict"][client_i])

        args.start_round = ckpt['round']
        global_encoders = ckpt['global_encoders']
        global_decoder_prior = ckpt['global_decoder_prior']
        agg_state = ckpt['agg_state']
        best_dices = ckpt['best_dices']

        print("load best result: {}".format(best_dices))

    ########## FL Training ##########
    for round in tqdm(range(args.start_round, args.c_rounds+1)):
        start = time.time()

        active_clients = select_clients(round, agg_state)
        logging.info('\n | Federated Round : {} | active clients: {} |'.format(round, [c+1 for c in active_clients]))

        ##### local training (parallel across available GPUs)
        local_encoders = {}
        local_decoders = {}
        result = []
        result_client_ids = []

        num_active = len(active_clients)
        branch_num = num_active // args.num_devices
        if branch_num % args.num_devices:
            branch_num += 1

        for branch in range(branch_num):

            torch.cuda.empty_cache()
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
            encoders, decoder, loss, m, optim = r.get()
            local_encoders[client_i] = encoders
            local_decoders[client_i] = decoder
            model_clients[client_i].load_state_dict(m.state_dict())
            optimizer_clients[client_i].load_state_dict(optim)

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

        broadcast_weights(model_clients, global_encoders, global_decoder_prior)

        ##### Eval the model after aggregation and every args.eval rounds
        if (round+1)%args.eval==0:
            logging.info('-'*20 + 'Test All the Models per 10 round'+ '-'*20)
            with torch.no_grad():
                results = []
                branch_num = args.client_num // args.num_devices
                if branch_num % args.num_devices:
                    branch_num += 1

                for branch in range(branch_num):

                    torch.cuda.empty_cache()
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
                    c_model = model_clients[c]
                    avgdice_score = sum(dice_score)/len(dice_score)
                    logging.info('--- Eval at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                                        .format((round), avgdice_score*100, dice_score))
                    writer.add_scalar('Eval_AvgDice/client_'+str(c+1), avgdice_score*100, round)

                    if best_dices[c] < avgdice_score:
                        best_dices[c] = avgdice_score
                        torch.save({
                            'round': round+1,
                            'dice': dice_score,
                            'state_dict': c_model.state_dict(),
                        }, args.modelfile_path + '/client-%d_round_%d_model_best.pth'%(c+1, round))

        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        if (round+1)%args.eval == 0:
            torch.save({

            'round': round + 1,

            "clients_dict": [model_clients[client_i].state_dict() for client_i in range(args.client_num)],
            "clients_optim_dict": [optimizer_clients[client_i].state_dict() for client_i in range(args.client_num)],

            'global_encoders': global_encoders,
            'global_decoder_prior': global_decoder_prior,
            'agg_state': agg_state,

            'best_dices': best_dices,
            }, args.modelfile_path + '/last.pth')

    writer.close()
