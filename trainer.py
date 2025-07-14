from typing import Tuple
import argparse
import time
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.MERIT import MERIT
from models.data.evaluation import cal_metrics
from models.data.dataloader import get_dataloader

from noter import Noter


class Trainer(object):
    def __init__(self,
                 args: argparse,
                 noter: Noter):
        print('[info] Loading data')
        self.trainloader, self.valloader, self.testloader = get_dataloader(args)
        self.n_user, self.n_item_a, self.n_item_b, self.n_item = args.n_user, args.n_item_a, args.n_item_b, args.n_item
        print('Done.\n')

        self.noter = noter
        self.device = args.device
        self.n_mtc = args.n_mtc

        # models
        self.model = MERIT(args).to(args.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self,
                  i_epoch: int,
                  ) -> Tuple[list, list]:
        self.model.train()
        self.optimizer.zero_grad()
        loss_f, loss_m = 0., 0.
        t0 = time.time()

        # training
        self.noter.log_msg(f'\n[epoch {i_epoch:>2}]')
        for batch in tqdm(self.trainloader, desc='training', leave=False):
            loss_f_batch, loss_m_batch = self.train_batch(batch)

            n_seq = batch[0].size(0)
            loss_f += (loss_f_batch * n_seq)
            loss_m += (loss_m_batch * n_seq)

        self.noter.log_train(loss_f / self.n_user, loss_m / self.n_user, time.time() - t0)

        # validating
        self.model.eval()
        res_ranks = [[] for _ in range(2)]
        with torch.no_grad():
            for batch in tqdm(self.valloader, desc='validating', leave=False):
                res_batch = self.evaluate_batch(batch)

                res_ranks = [res_set + list(res) for res_set, res in zip(res_ranks, res_batch)]

        return cal_metrics(res_ranks[0]), cal_metrics(res_ranks[1])

    def run_test(self,
                 ) -> list:
        self.model.eval()

        res_ranks = [[] for _ in range(2)]

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='testing', leave=False):
                res_batch = self.evaluate_batch(batch)

                res_ranks = [res_set + res for res_set, res in zip(res_ranks, res_batch)]

        res_rank = [cal_metrics(ranks) for ranks in res_ranks]

        return res_rank

    def train_batch(self,
                    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    ) -> Tuple[float, float]:
        seq_m, gt_m, gt_ab, gt_neg_m, gt_neg_ab = map(lambda x: x.to(self.device), batch)

        h_m, h_a, h_b = self.model(seq_m)

        loss_f = self.model.cal_rec_loss(h_a + h_b + h_m.detach(), gt_ab, gt_neg_ab)
        loss_m =  self.model.cal_rec_loss(h_m, gt_m, gt_neg_m)

        (loss_f + loss_m).backward()

        self.optimizer.step()
        return loss_f.item(), loss_m.item()

    def evaluate_batch(self,
                       batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                       ) -> Tuple[list, list]:
        seq_m, idx_last_a, idx_last_b, gt, gt_mtc = map(lambda x: x.to(self.device), batch)

        hs = self.model(seq_m, idx_last_a, idx_last_b)

        return self.model.cal_rank(hs, gt, gt_mtc)
