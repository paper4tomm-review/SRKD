import argparse
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    ######################## general settings ########################
    parser.add_argument('--output_dir', type=str, metavar='PATH', default='logs')
    parser.add_argument("--local_rank", type=int, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, help="world_size for distributed training")
    parser.add_argument('--eval_period', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_period', type=int, default=100)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--momentum_alpha', type=float, default=0.995)

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## MAE setting  ###################################
    parser.add_argument("--need_MAE", default=False, action='store_true',help="whether to use MAE")
    parser.add_argument("--mask_ratio", type=float, default=0.7, help="MAE image mask_ratio")
    parser.add_argument("--need_limit", default=False, action='store_true', help="whether to limit MAE")
    parser.add_argument("--need_pseudo", default=False, action='store_true', help="whether to limit pseudo")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+mlm+rkd+mkd+tri', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument("--rkd_loss_weight", type=float, default=1.0, help="kl loss weight")
    parser.add_argument("--mae_loss_weight", type=float, default=10, help="mae loss weight")
    parser.add_argument("--tri_loss_weight", type=float, default=10, help="tri loss weight")
    parser.add_argument("--mkd_loss_weight", type=float, default=1.0, help="mae kl loss weight")

    ######################## solver settings ########################
    parser.add_argument('--optimizer', type=str, default='Adam', help='[SGD, Adam, Adamw]')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    ######################## scheduler ########################
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))

    ######################## dataset ########################
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num_instance', type=int, default=4)
    parser.add_argument("--sampler", default="random", help="choose sampler from [identity, random, idrannum]")
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='/your/path')
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    # parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument("--test", dest='training', default=True, action='store_false')
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")

    args = parser.parse_args()

    return args