import logging
import os.path as op
import time
import torch
import torch.nn as nn
from reid.loss import objectives
from torch import amp

import torch.nn.functional as F
from reid.model.build import concat_all_gather
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from reid.utils.serialization import save_checkpoint
from prettytable import PrettyTable
import torch.distributed as dist

def do_train(start_epoch, num_epoch, args, model, old_model, train_loader, evaluator, optimizer,
             scheduler, set_index, writer, name, device):

    log_period = args.log_period
    eval_period = args.eval_period
    # device = "cuda"
    # device = torch.device("cuda:1")  # 选择 1 号 GPU
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("LTI-ReID.train")
    logger.info('current train dataset: {}'.format(name))
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mae_loss": AverageMeter(),
        "rkd_loss": AverageMeter(),
        "mkd_loss": AverageMeter(),
        "tri_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    # reduction: none, mean, sum, batchmean
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    current_task = [l.strip() for l in args.loss_names.split('+')]
    print(f'Training Model with {current_task} tasks')

    scaler = amp.GradScaler()
    # DDP
    world_size = dist.get_world_size()
    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = dict()
            images = batch['images']
            caption_ids = batch['caption_ids']
            if old_model is not None:
                with torch.no_grad():
                    # i_feats_old, t_feats_old, restore_feats, _ = old_model(batch)
                    i_feats_old, image_feats_old = old_model.module.encode_image(images, KL_Dis=True)
                    t_feats_old = old_model.module.encode_text(caption_ids)
                    # DDP
                    if dist.is_initialized():
                        dist.barrier()
                    i_feats_old = concat_all_gather(i_feats_old)  # 不带梯度的聚合
                    t_feats_old = concat_all_gather(t_feats_old)
                    image_feats_old = image_feats_old / image_feats_old.norm(dim=1, keepdim=True)
                    i_feats_old = i_feats_old / i_feats_old.norm(dim=1, keepdim=True)
                    t_feats_old = t_feats_old / t_feats_old.norm(dim=1, keepdim=True)

            with amp.autocast(device_type="cuda", enabled=True):
                i_feats, t_feats, restore_feats, mask, mret = model(batch, set_index)

            image_norm = i_feats / i_feats.norm(dim=1, keepdim=True)
            text_norm = t_feats / t_feats.norm(dim=1, keepdim=True)

            logit_scale = model.module.logit_scale
            ret.update({'temperature': 1 / logit_scale})

            if 'rkd' in current_task and set_index > 0:
                i2t_pred_new, t2i_pred_new = objectives.get_similarity_matrix(image_norm, text_norm, logit_scale)
                i2t_pred_old, t2i_pred_old = objectives.get_similarity_matrix(i_feats_old, t_feats_old, old_model.module.logit_scale)
                i2t_pred_new_log = torch.log(i2t_pred_new)
                t2i_pred_new_log = torch.log(t2i_pred_new)
                rkd_loss = 0.5 * kl_loss(i2t_pred_new_log, i2t_pred_old) + 0.5 * kl_loss(t2i_pred_new_log, t2i_pred_old)
                ret.update({'rkd_loss': rkd_loss * args.rkd_loss_weight})

            if 'mkd' in current_task and set_index > 0:
                restore_feats = restore_feats[:, 1:, :].float()  # remove cls token
                image_feats_old = image_feats_old[:, 1:, :].float()  # remove cls token
                restore_feats = F.normalize(restore_feats, dim=-1)  # (B, N, D)
                image_feats_old = F.normalize(image_feats_old, dim=-1)  # (B, N, D)

                mse_loss = ((restore_feats - image_feats_old) ** 2).mean(dim=-1)
                mkd_loss = (mse_loss * mask).sum() / mask.sum()

                # 计算余弦相似度损失
                # cosine_sim = (restore_feats * image_feats_old).sum(dim=-1)
                # mkd_loss = (1 - cosine_sim).mean()
                ret.update({'mkd_loss': mkd_loss * args.mkd_loss_weight})

            total_loss = sum([v for k, v in ret.items() if "loss" in k]) + sum([v for k, v in mret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['rkd_loss'].update(ret.get('rkd_loss', 0), batch_size)
            meters['mkd_loss'].update(ret.get('mkd_loss', 0), batch_size)
            meters['sdm_loss'].update(mret.get('sdm_loss', 0), batch_size)
            meters['mae_loss'].update(mret.get('mae_loss', 0), batch_size)
            meters['tri_loss'].update(mret.get('tri_loss', 0), batch_size)
            meters['mlm_loss'].update(mret.get('mlm_loss', 0), batch_size)
            meters['mlm_acc'].update(mret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('temperature', ret['temperature'], epoch)

        for k, v in meters.items():
            if v.avg > 0:
                writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            torch.cuda.empty_cache()  # 评估前清理显存
            # if get_rank() == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            with torch.no_grad():
                if args.distributed:
                    top1, mAP, _ = evaluator.eval(model.module.eval())
                else:
                    top1, mAP, _ = evaluator.eval(model.eval())

            # 每个 epoch 保存模型检查点
            # if dist.get_rank() == 0:
                # torch.save(model.module.state_dict(), f'checkpoint_epoch_{epoch}.pt')

            torch.cuda.empty_cache()
            if best_top1 < top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                # save_name = '{}_checkpoint.pth.tar'.format(name)
                # save_checkpoint({
                #     'state_dict': model.state_dict(),
                #     'epoch': epoch,
                #     'mAP': mAP,
                # }, True, fpath=op.join(args.output_dir, save_name))

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")



def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("SRKD.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1, _, _ = evaluator.eval(model.eval())
