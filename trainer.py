from argparse import Namespace
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
                 args: Namespace,
                 noter: Noter
                 )-> None:
        print('[info] Loading data')
        self.n_warmup = args.n_warmup
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.n_user = args.n_user
        self.n_item_a = args.n_item_a
        print('Done.\n')

        self.noter = noter
        self.device = args.device

        # models
        self.model = MERIT(args).to(args.device)

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self,
                  i_epoch: int,
                  ) -> tuple[list | None, list | None]:
        self.model.train()
        loss_a, loss_b, loss_m = 0., 0., 0.
        t_0 = time.time()

        # training
        for batch in tqdm(self.train_loader, desc='training', leave=False):
            self.optimizer.zero_grad()

            loss_a_batch, loss_b_batch, loss_m_batch = self.train_batch(batch)

            n_seq = batch[0].size(0)
            loss_a += (loss_a_batch * n_seq) / self.n_user
            loss_b += (loss_b_batch * n_seq) / self.n_user
            loss_m += (loss_m_batch * n_seq) / self.n_user

        self.noter.log_train(i_epoch, loss_a, loss_b, loss_m, time.time() - t_0)

        # warm-up quit
        if i_epoch <= self.n_warmup:
            return None, None

        # validating
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='validating', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def run_test(self,
                 ) -> tuple[list, list]:
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='testing', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def train_batch(self,
                    batch: list[torch.Tensor],
                    ) -> tuple[float, float, float]:
        seq_m, gt_m, gt_ab, gt_neg_m, gt_neg_ab = map(lambda x: x.to(self.device), batch)

        h_m, h_a, h_b = self.model(seq_m)

        loss_a, loss_b = self.model.cal_rec_loss(h_a + h_b + h_m.detach(), gt_ab, gt_neg_ab)
        loss_ma, loss_mb =  self.model.cal_rec_loss(h_m, gt_m, gt_neg_m)
        loss_m = loss_ma + loss_mb

        (loss_a + loss_b + loss_m).backward()

        self.optimizer.step()
        return loss_a.item(), loss_b.item(), loss_m.item()

    def evaluate_batch(self,
                       batch: list[torch.Tensor],
                       ) -> tuple[list[float], list[float]]:
        seq_m, idx_last_a, idx_last_b, gt, gt_mtc = map(lambda x: x.to(self.device), batch)

        h_m, h_a, h_b = self.model(seq_m, idx_last_a, idx_last_b)

        ranks, mask_gt_a, mask_gt_b = self.model.cal_rank(h_m, h_a, h_b, gt, gt_mtc)
        ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
        ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()

        return ranks_a, ranks_b
