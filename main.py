import argparse
import os
import random
from os.path import join
import numpy as np
import torch

from noter import Noter
from trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description='MERIT-Experiment')
    parser.add_argument('--name', type=str, default='MERIT (MM\'25)', help='name of the models')
    parser.add_argument('--ver', type=str, default='0', help='development notes')
    parser.add_argument('--data', type=str, default='abe', help='afk: Food-Kitchen'
                                                                'abe: Beauty-Electronics'
                                                                'amb: Movie-Book')

    # Data
    parser.add_argument('--raw', action='store_true', help='use raw data generated by preprocess.py')
    parser.add_argument('--len_max', type=int, default=50, help='# of interactions allowed to input')
    parser.add_argument('--n_neg', type=int, default=128, help='# negative inference samples')
    parser.add_argument('--n_mtc', type=int, default=999, help='# negative metric samples')

    # Model
    parser.add_argument('--d_embed', type=int, default=256, help='dimension of latent representation')
    parser.add_argument('--n_attn', type=int, default=1, help='# layer of TransformerEncoderLayer stack')
    parser.add_argument('--n_head', type=int, default=2, help='# multi-head for self-attention')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--temp', type=float, default=0.75, help='temperature for InfoNCE')

    # Training
    parser.add_argument('--cuda', type=str, default='0', help='running device')
    parser.add_argument('--seed', type=int, default=3407, help='random seeding')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--n_worker', type=int, default=0, help='# dataloader worker')
    parser.add_argument('--n_epoch', type=int, default=500, help='# epoch maximum')
    parser.add_argument('--n_warmup', type=int, default=10, help='# warmup epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e0, help='weight decay')
    parser.add_argument('--lr_g', type=float, default=0.5, help='scheduler gamma')
    parser.add_argument('--lr_p', type=int, default=30, help='scheduler patience')

    args = parser.parse_args()

    if args.n_warmup >= args.n_epoch:
        args.n_warmup = max(0, args.n_epoch - 1)

    if args.cuda == 'cpu':
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.cuda}')

    args.len_trim = args.len_max - 3  # leave-one-out
    args.es_p = (args.lr_p + 1) * 2 - 1
    args.bse = args.bs * 4

    # paths
    args.path_root = os.getcwd()
    args.path_data = join(args.path_root, 'data', args.data)
    args.path_log = join(args.path_root, 'log')
    for p in [args.path_data, args.path_log]:
        if not os.path.exists(p):
            os.makedirs(p)

    args.f_raw = join(args.path_data, args.data + f'_{args.len_max}_preprocessed.txt')
    args.f_data = join(args.path_data, args.data + f'_{args.len_max}_seq.pkl')

    if args.raw and not os.path.exists(args.f_raw):
        raise FileNotFoundError(f'Selected preprocessed dataset {args.data} does not exist.')
    if not args.raw and not os.path.exists(args.f_data):
        if os.path.exists(args.f_raw):
            raise FileNotFoundError(f'Selected dataset {args.data} need process, specify "--raw" in the first run.')
        raise FileNotFoundError(f'Selected processed dataset {args.data} does not exist.')

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # modeling
    noter = Noter(args)
    trainer = Trainer(args, noter)

    cnt_es, cnt_lr, mrr_log = 0, 0, 0.
    res_a, res_b = [0], [0]

    for epoch in range(1, args.n_epoch + 1):
        lr_cur = trainer.optimizer.param_groups[0]["lr"]
        res_val_a, res_val_b = trainer.run_epoch(epoch)
        mrr_val = res_val_b[-1] + res_val_a[-1]  # use mrr as identifier
        noter.log_valid(res_val_a, res_val_b)

        if epoch <= args.n_warmup:
            lr_str = f'{lr_cur:.5e}'
            noter.log_lr(f'| {lr_str[:3]}e-{lr_str[-1]} | warmup |')
            trainer.scheduler_warmup.step()

        else:
            if mrr_val >= mrr_log:
                mrr_log = mrr_val
                cnt_es = 0
                cnt_lr = 0
                lr_str = f'{lr_cur:.5e}'
                noter.log_lr(f'| {lr_str[:3]}e-{lr_str[-1]} |  0 /{args.lr_p:2} |  0 /{args.es_p:2} |')

                res_a, res_b = trainer.run_test()

                noter.log_test(res_a, res_b)

                trainer.scheduler.step(epoch)

            else:
                cnt_lr += 1
                cnt_es += 1
                if cnt_es > args.es_p:
                    noter.log_msg(f'\n[info] Exceeds maximum early-stop patience.')
                    break
                else:
                    trainer.scheduler.step(0)

                    lr_str = f'{lr_cur:.5e}'
                    noter.log_msg(f'|     | {lr_str[:3]}e-{lr_str[-1]} | {cnt_lr:2} /{args.lr_p:2} | {cnt_es:2} /{args.es_p:2} |')
                    if lr_cur != trainer.optimizer.param_groups[0]["lr"]:
                        cnt_lr = 0

    noter.log_final_result(res_a, res_b)


if __name__ == '__main__':
    main()
