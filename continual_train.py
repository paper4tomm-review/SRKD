import os
import os.path as op
import numpy as np
import sys
import time
import copy
import torch
import random
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import collections
import torch.nn.functional as F

from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from torch.utils.tensorboard import SummaryWriter

from datasets.build_loaders import build_data_loaders
from reid.processor.processor import do_train
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from reid.solver import build_optimizer, build_lr_scheduler
from reid.model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed):
    random.seed(seed)                   
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)     
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main_worker(args):
    local_rank = int(os.environ["LOCAL_RANK"]) 
    args.distributed = True
    args.local_rank = local_rank

    if args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method="env://")

    world_size = torch.distributed.get_world_size()
    args.world_size = world_size

    seed = args.seed + dist.get_rank()
    set_seed(seed)

    # set output log
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda", local_rank)
    
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, f'{cur_time}')
    logger = setup_logger('LTI-ReID', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['cuhkpedes', 'icfgpedes', 'rstpreid', 'iiitd']
    else:
        training_set = ['iiitd', 'cuhkpedes', 'rstpreid', 'icfgpedes']
    # all the revelent datasets
    all_set = ['cuhkpedes', 'icfgpedes', 'rstpreid', 'iiitd', 'ufine3c']
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the dataloader of all datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)

    first_train_set = all_train_sets[0]
    model = build_model(args, num_classes=first_train_set[1])
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

    writer = SummaryWriter(log_dir=args.output_dir)
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(op.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)     #
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model
            checkpoint = load_checkpoint(op.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

            model = linear_combination(args, model, model_old, 0.5)

            save_name = '{}_checkpoint_adaptive_ema.pth.tar'.format(training_set[step+1])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=op.join(args.output_dir, save_name))
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1, logger)

        exit(0)

    resume_epoch = 0
    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        resume_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(resume_epoch, best_mAP))

    # train on the datasets squentially
    for set_index in range(0, len(training_set)):
        model, old_model = train_dataset(resume_epoch, args, all_train_sets, set_index, model, writer, logger, device)
        if set_index > 0:
            model = linear_combination(model, old_model, 0.5)
        test_model(model, all_train_sets, all_test_only_sets, set_index, logger)
    print('finished')


def train_dataset(resume_epoch, args, all_train_sets, set_index, model, writer, logger, device):
    # status of current dataset
    dataset, num_classes, train_loader, val_img_loader, val_txt_loader, init_loader, name = all_train_sets[set_index]

    args.num_epoch = 60 if set_index == 0 else 60
    print("current epoch:", args.num_epoch)
    num_epoch = args.num_epoch

    old_model = None

    if set_index > 0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

    # Re-initialize optimizer
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # set evaluator
    evaluator = Evaluator(val_img_loader, val_txt_loader)
    start_epoch = (resume_epoch + 1) if args.resume and set_index == 0 else 1

    # model training
    do_train(start_epoch, num_epoch, args, model, old_model, train_loader, evaluator, optimizer,
             scheduler, set_index, writer, name, device)

    return model, old_model


def test_model(model, all_train_sets, all_test_sets, set_index, logger):

    logger = logging.getLogger("LTI-ReID.test")
    R1_all = []
    mAP_all = []
    mINP_all = []
    # process on seen dataset
    logger.info(f"Processing seen datasets up to index {set_index}")
    for i in range(0, set_index + 1):
        _, _, _, val_img_loader, val_txt_loader, _, name = all_train_sets[i]
        logger.info('Results on {}'.format(name))
        evaluator = Evaluator(val_img_loader, val_txt_loader)
        # if get_rank() == 0:
        with torch.no_grad():
            if args.distributed:
                train_R1, train_mAP, train_mINP = evaluator.eval(model.module.eval())
            else:
                train_R1, train_mAP, train_mINP = evaluator.eval(model.eval())
            del evaluator  
            torch.cuda.empty_cache()
        
        if args.distributed:
            dist.barrier()

        R1_all.append(float(train_R1))
        mAP_all.append(float(train_mAP))
        mINP_all.append(float(train_mINP))

    aver_R1 = torch.tensor(R1_all).mean()
    aver_mAP = torch.tensor(mAP_all).mean()
    aver_mINP = torch.tensor(mINP_all).mean()

    logger.info("Average R1 on Seen dataset: {:.1f}%".format(aver_R1))
    logger.info("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP))
    logger.info("Average mINP on Seen dataset: {:.1f}%".format(aver_mINP))

    # process on unseen dataset
    R1_all = []
    mAP_all = []
    mINP_all = []

    for i in range(len(all_test_sets)):
        _, _, test_img_loader, test_txt_loader, name = all_test_sets[i]
        logger.info('Results on {}'.format(name))
        evaluator = Evaluator(test_img_loader, test_txt_loader)
        if get_rank() == 0:
            with torch.no_grad():
                if args.distributed:
                    R1, mAP, mINP = evaluator.eval(model.module.eval())
                else:
                    R1, mAP, mINP = evaluator.eval(model.eval())
                del evaluator
                torch.cuda.empty_cache()
        else:
            R1, mAP, mINP = 0.0, 0.0, 0.0

        if args.distributed:
            dist.barrier()

        R1_all.append(float(R1))
        mAP_all.append(float(mAP))
        mINP_all.append(float(mINP))

    aver_R1_unseen = torch.tensor(R1_all).mean()
    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_mINP_unseen = torch.tensor(mINP_all).mean()

    logger.info("Average R1 on Unseen dataset: {:.1f}%".format(aver_R1_unseen))
    logger.info("Average mAP on Unseen dataset: {:.1f}%".format(aver_mAP_unseen))
    logger.info("Average mINP on Unseen dataset: {:.1f}%".format(aver_mINP_unseen))


def linear_combination(model, model_old, alpha):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':

    args = get_args()
    main_worker(args)
