from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from reid.loss import objectives
import torch.distributed as dist
from .pos_embed import get_2d_sincos_pos_embed

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.patch_size= base_cfg['vision_patch_size']
        self.grid_size = (args.img_size[0] // self.patch_size, args.img_size[1] // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'mkd' in args.loss_names:
            self.norm_pix_loss=False
            self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.cross_attn_Text = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer_mae = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            self.norm_1=LayerNorm(self.embed_dim)
            self.norm_2=LayerNorm(self.embed_dim)
            self.norm_3=LayerNorm(self.embed_dim)
            self.norm_4=LayerNorm(self.embed_dim)
            self.norm_5=LayerNorm(self.embed_dim)
            self.norm_6=LayerNorm(self.embed_dim)
            self.norm_7=LayerNorm(self.embed_dim)

            #参数初始化
            scale = self.cross_modal_transformer_mae.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer_mae.layers)**-0.5)
            attn_std = scale
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            nn.init.normal_(self.cross_attn_Text.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_Text.out_proj.weight, std=proj_std)

            fc_std = (2 * self.cross_modal_transformer_mae.width)**-0.5
            for block in self.cross_modal_transformer_mae.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        # print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def cross_former_mae(self, image_feats, text_feats, ids_restore):
        image_feats=self.norm_1(image_feats)
        text_feats=self.norm_2(text_feats)

        x = self.decoder_embed(image_feats)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # text_feats = torch.nn.functional.pad(text_feats, (0, 0, 0, x.shape[1]-text_feats.shape[1]))
        # x = (x+text_feats)/2
        x = x + self.decoder_pos_embed
        x = x.to(torch.float16)
        need_text = self.cross_attn_Text(
                self.norm_4(x),
                text_feats,
                text_feats,
                need_weights=False)[0]
        # x = self.norm_6(x + need_text)
        x = self.norm_6(need_text)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_mae(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        restore_feats = self.norm_7(x)

        return restore_feats

    def encode_image(self, image, KL_Dis=False):
        x = self.base_model.encode_image(image)
        if KL_Dis == False:
            return x[:, 0, :].float()
        else:
            return x[:, 0, :].float(), x.float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def reorder_second_group(self, second_group, similarity_matrix, label_matrix):
        similarity_matrix[label_matrix == 1] = float('-inf')
        max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
        reordered_second_group = second_group[max_indices]
        return reordered_second_group

    def rgb_to_weighted_grayscale(self, rgb_tensor):
        r_weight = 0.2989
        g_weight = 0.5870
        b_weight = 0.1140
        device = rgb_tensor.device
        weights = torch.tensor([r_weight, g_weight, b_weight], device=device)
        grayscale_tensor = torch.sum(rgb_tensor * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        grayscale_tensor = grayscale_tensor.repeat(1, 3, 1, 1)
        return grayscale_tensor

    def forward(self, batch, set_index):
        mret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        if 'sdm' in self.current_task:
            if dist.is_initialized():
                dist.barrier()
            i_feats_all = all_gather_with_grad(i_feats)
            t_feats_all = all_gather_with_grad(t_feats)
            labels_all = concat_all_gather(batch['pids'])
            sdm_loss, result_sdm, labels = objectives.compute_sdm(i_feats_all, t_feats_all, labels_all, self.logit_scale)
            mret.update({'sdm_loss':sdm_loss})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            mret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            mret.update({'img_acc': image_precision})
            mret.update({'txt_acc': text_precision})

        if 'tri' in self.current_task:
            margin = 0.05
            ne_i_feats = self.reorder_second_group(i_feats_all, result_sdm, labels)
            ne_t_feats = self.reorder_second_group(t_feats_all, result_sdm.t(), labels.t())
            loss1 = objectives.compute_tri(t_feats_all, i_feats_all, ne_i_feats, margin)
            loss2 = objectives.compute_tri(i_feats_all, t_feats_all, ne_t_feats, margin)
            tri_loss = loss1 + loss2
            mret.update({'tri_loss': self.args.tri_loss_weight * tri_loss})

        if 'mlm' in self.current_task:
            # mlm_ids = batch['mlm_ids']
            # mlm_feats = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(text_feats, image_feats, image_feats)
            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            mlm_loss = objectives.compute_mlm(scores, mlm_labels)
            mret.update({'mlm_loss': mlm_loss * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            mret.update({'mlm_acc': acc})

        if set_index == 0:
            restore_feats = None
            mask = None

        if 'mkd' in self.current_task and set_index > 0:
            grayscale_images = self.rgb_to_weighted_grayscale(images)
            mae_image_feats, mae_text_feats, mask, ids_restore = self.base_model(grayscale_images, caption_ids, need_MAE=self.args.need_MAE, mask_ratio=self.args.mask_ratio)
            # mae_image_feats, mask, ids_restore = self.base_model.encode_image(grayscale_images, need_mae=self.args.need_MAE, mask_ratio=self.args.mask_ratio)
            restore_feats = self.cross_former_mae(mae_image_feats, mae_text_feats, ids_restore)

        return i_feats_all, t_feats_all, restore_feats, mask, mret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    # convert_weights(model)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor = tensor.contiguous()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      # rank_num * tensor_size


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        ctx.world_size = torch.distributed.get_world_size()
        output = [torch.zeros_like(x) for _ in range(ctx.world_size)]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # 合并所有梯度并分割到各进程
        all_gradients = torch.cat(grads, dim=0)
        world_size = ctx.world_size
        batch_size = all_gradients.size(0) // world_size
        rank = torch.distributed.get_rank()
        local_grad = all_gradients[rank * batch_size : (rank + 1) * batch_size]
        return local_grad

def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)