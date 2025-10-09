import copy
import os.path
import os
from reid.utils.feature_tools import *
import datasets as datasets

import numpy as np
import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler, RandomIdnumSampler
from .sampler_ddp import RandomIdentitySampler_DDP, DistributedRandomIdnumSampler
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset, ImageTextMAEDataset


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def get_data(args, name, training=False):
    logger = logging.getLogger("LTI-ReID.dataset")

    root = args.data_dir
    num_workers = args.workers
    dataset = datasets.create(name, root)
    if training:
        num_classes = len(dataset.train_id_container)
    else:
        num_classes = len(dataset.test_id_container)

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]


    if training:
        height, width = args.img_size

        train_transforms = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])

        val_transforms = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                            train_transforms,
                                            text_length=args.text_length)
        elif args.need_MAE:
            train_set = ImageTextMAEDataset(dataset.train,
                                            train_transforms,
                                            text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)

        init_set = ImageTextDataset(dataset.train,
                                         val_transforms,
                                         text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                # pin_memory=True  # 可选，加速数据加载
                train_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate, pin_memory=True)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'idrannum':
            if args.distributed:
                logger.info('DISTRIBUTED TRAIN START')
                logger.info('using idrannum sampler')
                sampler = DistributedRandomIdnumSampler(
                    dataset.train, args.batch_size, args.num_instance
                )
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    collate_fn=collate,
                    drop_last=True,
                    pin_memory=True
                )
            else:
                logger.info('using idrannum sampler')
                train_loader = DataLoader(train_set,
                                            batch_size=args.batch_size,
                                            sampler=RandomIdnumSampler(
                                                dataset.train, args.batch_size,
                                                args.num_instance),
                                            num_workers=num_workers,
                                            collate_fn=collate)
        elif args.sampler == 'random':
            if args.distributed:
                # 分布式随机采样逻辑
                logger.info('DISTRIBUTED TRAIN START')
                logger.info('using random sampler')
                train_sampler = DistributedSampler(train_set)
                train_loader = DataLoader(
                    train_set,
                    pin_memory=True,
                    batch_size=args.batch_size,
                    sampler=train_sampler,
                    shuffle=False,  # DistributedSampler会自行处理shuffle
                    num_workers=num_workers,
                    collate_fn=collate
                )
            else:
                # TODO add distributed condition
                logger.info('using random sampler')
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        init_loader = DataLoader(init_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False)

        return [dataset, num_classes, train_loader, val_img_loader, val_txt_loader, init_loader, name]
    else:
        # build dataloader for testing
        height, width = args.img_size
        test_transforms = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        return [dataset, num_classes, test_img_loader, test_txt_loader, name]




def build_data_loaders(args, training_set, testing_only_set):
    # Create data loaders
    training_loaders = [get_data(args, name, training=True) for name in training_set]
    testing_loaders = [get_data(args, name, training=False) for name in testing_only_set]
    return training_loaders, testing_loaders
