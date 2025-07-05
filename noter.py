import os
from os.path import join
import time

import argparse
import torch.nn as nn


class Noter(object):
    """ console printing and saving into files """
    def __init__(self,
                 args: argparse,
                 ) -> None:
        self.args = args

        self.t_start = time.time()
        self.f_log = join(args.path_log, f'{args.data}-{args.len_max}-{time.strftime("%m-%d-%H-%M", time.localtime())}-'
                                         f'{str(args.device)[0] + str(args.device)[-1]}-{args.seed}-MERIT.log')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove duplicate

        self.n_param = 0

        # welcome
        self.log_msg(f'\n{"-" * 30} Experiment {self.args.name} {"-" * 30}')
        self.log_settings()

    def write(self,
              msg: str,
              ) -> None:
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    def log_msg(self,
                msg: str,
                ) -> None:
        print(msg)
        self.write(msg)

    def log_settings(self,
                     ) -> None:
        msg = (f'[Info] {self.args.name} (data:{self.args.data}, cuda:{self.args.cuda})\n'
               f'| Ver.  {self.args.ver} |\n'
               f'| n_attn {self.args.n_attn} | n_head {self.args.n_head} | dropout {self.args.dropout} |\n'
               f'| lr {self.args.lr:.2e} | l2 {self.args.l2:.2e} | lr_g {self.args.lr_g:.1f} | lr_p {self.args.lr_p} |\n'
               f'| {self.args.data}-{self.args.len_max} | seed {self.args.seed} |\n')
        self.log_msg(msg)

    def log_lr(self,
               msg: str
               ) -> None:
        msg = f'           | lr  |     ' + msg
        self.log_msg(msg)

    def log_num_param(self,
                      model: nn.Module = None,
                      ) -> None:
        if self.n_param == 0 and model is not None:
            self.n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_msg(f'[info] model contains {self.n_param} learnable parameters.\n')

    def log_train(self,
                  loss_f: float,
                  loss_m: float,
                  t_gap: float,
                  ) -> None:
        msg = f'| tr  | {f"{loss_f:.4f}"[:6]} | {f"{loss_m:.4f}"[:6]} | {t_gap:>5.1f}s |'
        self.log_msg(msg)

    def log_valid(self,
                  res_a: list,
                  res_b: list,
                  ) -> None:
        msg = (f'| val | {res_a[0]:.4f} | {res_a[1]:.4f} | {res_a[2]:.4f} | {res_a[3]:.4f} |'
               f' {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f} | {res_b[3]:.4f} |')
        self.log_msg(msg)

    def log_test(self,
                 res_a: list,
                 res_b: list,
                 ) -> None:
        msg = f'| te  | {res_a[0]:.4f} | {res_a[1]:.4f} | {res_a[2]:.4f} | {res_a[3]:.4f} | {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f} | {res_b[3]:.4f} |'
        self.log_msg(msg)

    def log_final_result(self,
                         res_a: list,
                         res_b: list,
                         ) -> None:
        self.log_msg(f'\n{"-" * 10} Experiment ended {"-" * 10}')
        self.log_settings()
        msg = (f'[Info] {self.args.name} ({(time.time() - self.t_start) / 60:.1f} min)\n'
               f'|                A                  |                B                  |\n'
               f'|  hr5   |  hr10  | ndcg10 |  mrr   |  hr5   |  hr10  | ndcg10 |  mrr   |\n'
               f'| {res_a[0]:.4f} | {res_a[1]:.4f} | {res_a[2]:.4f} | {res_a[3]:.4f} | {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f} | {res_b[3]:.4f} |\n')
        self.log_msg(msg)
        self.log_num_param()
