from typing import Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention, CrossAttention2, CrossAttention
from models.ffn import MoFFN, FeedForward

from models.utils.initialization import init_weights
from models.utils.position import get_absolute_pos_idx
from models.data.evaluation import cal_norm_mask


class MERIT(torch.nn.Module):
    def __init__(self,
                 args: argparse,
                 ) -> None:
        super().__init__()
        self.bs = args.bs
        self.len_trim = args.len_trim
        self.n_neg = args.n_neg
        self.d_embed = args.d_embed
        self.temp = args.temp
        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else nn.Identity()

        self.n_item = args.n_item
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b

        # ----------------------------------
        # embedding
        self.ei = nn.Embedding(args.n_item + 1, args.d_embed, padding_idx=0)
        self.ep = nn.Embedding(args.len_trim + 1, args.d_embed, padding_idx=0)
        self.drop_emb = nn.Dropout(args.dropout)

        # ----------------------------------
        # self-attention encoder
        self.sa_m = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.sa_a = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.sa_b = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)

        # ----------------------------------
        #  moFFN
        self.ffn_m = MoFFN(args.d_embed, args.dropout)
        self.ffn_a = MoFFN(args.d_embed, args.dropout)
        self.ffn_b = MoFFN(args.d_embed, args.dropout)

        # ----------------------------------
        # ECAF_a and ECAF_b
        self.caf_a = CrossAttention2(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.caf_b = CrossAttention2(args.d_embed, args.n_head, args.len_trim, args.dropout)

        self.ffn_caf_a = FeedForward(args.d_embed, args.dropout)
        self.ffn_caf_b = FeedForward(args.d_embed, args.dropout)

        # ----------------------------------
        # CAF_m
        self.caf_m = CrossAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)

        self.ffn_caf_m = FeedForward(args.d_embed, args.dropout)

        self.apply(init_weights)

    def forward(self,
                seq_m: torch.Tensor,
                idx_last_a: torch.Tensor=None,
                idx_last_b: torch.Tensor=None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ----------------------------------
        # embedding
        mask_m = (seq_m != 0).int()
        mask_a = (seq_m > self.n_item_a).int()
        mask_b = ((seq_m > 0) & (seq_m <= self.n_item_a)).int()

        h_m = self.ei(seq_m)
        h_a = h_m + self.ep(get_absolute_pos_idx(mask_m))
        h_b = h_m + self.ep(get_absolute_pos_idx(mask_a))
        h_m = h_m + self.ep(get_absolute_pos_idx(mask_b))

        mask_m.unsqueeze_(-1)
        mask_a.unsqueeze_(-1)
        mask_b.unsqueeze_(-1)

        h_m = self.dropout(h_m * mask_m)
        h_a = self.dropout(h_a * mask_a)
        h_b = self.dropout(h_b * mask_b)

        # ----------------------------------
        # multi-head self-attention
        h_m = self.sa_m(h_m, mask_m)
        h_a = self.sa_a(h_a, mask_a)
        h_b = self.sa_b(h_b, mask_b)

        # ----------------------------------
        # moffn
        h_m, h_m2a, h_m2b = self.ffn_m(h_m, mask_m)
        h_a, h_a2m, h_a2b = self.ffn_a(h_a, mask_a)
        h_b, h_b2m, h_b2a = self.ffn_b(h_b, mask_b)

        # ----------------------------------
        # extended cross-attn fusion of A and B
        h_a = self.caf_a(h_a, h_m2a, h_b2a, mask_a)
        h_b = self.caf_b(h_b, h_m2b, h_a2b, mask_b)

        h_a = self.ffn_caf_a(h_a, mask_a)
        h_b = self.ffn_caf_b(h_b, mask_b)

        # ----------------------------------
        # cross-attn fusion for M
        h_m = self.caf_m(h_m, h_a2m + h_b2m, mask_m)

        h_m = self.ffn_caf_m(h_m, mask_m)

        # ----------------------------------
        # output
        if not self.training:
            idx_batched = torch.arange(h_a.size(0))
            h_m = h_m[:, -1]
            h_a = h_a[idx_batched, idx_last_a.squeeze(-1)]
            h_b = h_b[idx_batched, idx_last_b.squeeze(-1)]

        return h_m, h_a, h_b

    def cal_rec_loss(self,
                     h: torch.Tensor,
                     gt: torch.Tensor,
                     gt_neg: torch.Tensor,
                     mask: torch.Tensor = None,
                     ) -> torch.Tensor:
        """ InfoNCE """
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg)

        logits = torch.cat(((h * e_gt).unsqueeze(-2).sum(-1),
                            (h.unsqueeze(-2) * e_neg).sum(-1)), dim=-1).div(self.temp)

        mask_gt_a = torch.where(gt.gt(0) & gt.le(self.n_item_a), 1, 0)
        mask_gt_b = torch.where(gt.gt(self.n_item_a), 1, 0)

        if mask is not None:
            mask = mask.squeeze(-1)
            mask_gt_a *= mask
            mask_gt_b *= mask

        loss = -F.log_softmax(logits, dim=2)[:, :, 0]
        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()
        return loss_a + loss_b

    @staticmethod
    def cal_domain_rank(h: torch.Tensor,
                        e_gt: torch.Tensor,
                        e_mtc: torch.Tensor,
                        mask_gt_a: torch.Tensor,
                        mask_gt_b: torch.Tensor,
                        ) -> Tuple[list, list]:
        """ calculate domain rank via inner-product similarity """
        logit_gt = (h * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logit_mtc = (h.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logit_mtc - logit_gt).gt(0).sum(-1).add(1)

        return ranks[mask_gt_a == 1].tolist(), ranks[mask_gt_b == 1].tolist()

    def cal_rank(self,
                 hs: torch.Tensor,
                 gt: torch.Tensor,
                 gt_mtc: torch.Tensor,
                 ) -> Tuple[list, list]:
        """ rank via inner-product similarity """
        mask_gt_a = torch.where(gt <= self.n_item_a, 1, 0)
        mask_gt_b = torch.where(gt > self.n_item_a, 1, 0)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        (h_m, h_a, h_b) = hs
        h_f = h_a * mask_gt_a + h_b * mask_gt_b + h_m

        return self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a.squeeze(-1), mask_gt_b.squeeze(-1))
