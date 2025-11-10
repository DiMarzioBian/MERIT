from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention, CrossAttention, CrossAttention2
from models.ffn import MoFFN, FeedForward

from models.utils.initialization import init_weights
from models.utils.position import get_absolute_pos_idx
from models.data.evaluation import cal_norm_mask


class MERIT(torch.nn.Module):
    def __init__(self,
                 args: Namespace,
                 ) -> None:
        super().__init__()
        self.bs = args.bs
        self.len_trim: int = args.len_trim
        self.n_item: int = args.n_item
        self.n_item_a: int = args.n_item_a
        self.n_neg: int = args.n_neg
        self.temp: float = args.temp
        self.d_embed: int = args.d_embed

        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else nn.Identity()

        # embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # self-attention encoder
        self.sa_m = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.sa_a = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.sa_b = SelfAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)

        #  moFFN
        self.ffn_m = MoFFN(args.d_embed, args.dropout)
        self.ffn_a = MoFFN(args.d_embed, args.dropout)
        self.ffn_b = MoFFN(args.d_embed, args.dropout)

        # CAF_m
        self.caf_m = CrossAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)

        self.ffn_caf_m = FeedForward(args.d_embed, args.dropout)

        # ECAF_a and ECAF_b
        self.caf_a = CrossAttention2(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.caf_b = CrossAttention2(args.d_embed, args.n_head, args.len_trim, args.dropout)

        self.ffn_caf_a = FeedForward(args.d_embed, args.dropout)
        self.ffn_caf_b = FeedForward(args.d_embed, args.dropout)

        self.apply(init_weights)

    def embed_pos(self,
                  mask: torch.Tensor,
                  ) -> torch.Tensor:
        return self.ep(get_absolute_pos_idx(mask))

    def forward(self,
                seq_m: torch.Tensor,
                idx_last_a: torch.Tensor=None,
                idx_last_b: torch.Tensor=None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # embedding
        mask_m = (seq_m != 0).to(torch.int32).unsqueeze(-1)
        mask_a = (seq_m > self.n_item_a).to(torch.int32).unsqueeze(-1)
        mask_b = ((seq_m > 0) & (seq_m <= self.n_item_a)).to(torch.int32).unsqueeze(-1)

        h_m = self.ei(seq_m)
        h_a = self.dropout(h_m + self.embed_pos(mask_m)) * mask_m
        h_b = self.dropout(h_m + self.embed_pos(mask_m)) * mask_a
        h_m = self.dropout(h_m + self.embed_pos(mask_b)) * mask_b

        # multi-head self-attention
        h_m = self.sa_m(h_m, mask_m)
        h_a = self.sa_a(h_a, mask_a)
        h_b = self.sa_b(h_b, mask_b)

        # moFFN
        h_m, h_m2a, h_m2b = self.ffn_m(h_m, mask_m)
        h_a, h_a2m, h_a2b = self.ffn_a(h_a, mask_a)
        h_b, h_b2m, h_b2a = self.ffn_b(h_b, mask_b)

        # CAF_m
        h_m = self.caf_m(h_m, h_a2m + h_b2m, mask_m)
        h_m = self.ffn_caf_m(h_m, mask_m)

        # ECAF_a and ECAF_b
        h_a = self.caf_a(h_a, h_m2a, h_b2a, mask_a)
        h_a = self.ffn_caf_a(h_a, mask_a)

        h_b = self.caf_b(h_b, h_m2b, h_a2b, mask_b)
        h_b = self.ffn_caf_b(h_b, mask_b)

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
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_gt_a = torch.where(gt.gt(0) & gt.le(self.n_item_a), 1, 0)
        mask_gt_b = torch.where(gt.gt(self.n_item_a), 1, 0)

        e_gt = self.ei(gt).unsqueeze(-2)
        e_neg = self.ei(gt_neg)
        e_all = torch.cat([e_gt, e_neg], dim=-2)

        logits = torch.einsum('bld,blnd->bln', h, e_all).div(self.temp)

        loss = -F.log_softmax(logits, dim=2)[:, :, 0]

        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()

        return loss_a, loss_b

    def cal_rank(self,
                 h_m: torch.Tensor,
                 h_a: torch.Tensor,
                 h_b: torch.Tensor,
                 gt: torch.Tensor,
                 gt_mtc: torch.Tensor,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ rank via inner-product similarity """
        mask_gt_a = torch.where(gt <= self.n_item_a, 1, 0)
        mask_gt_b = torch.where(gt > self.n_item_a, 1, 0)
        h = h_a * mask_gt_a + h_b * mask_gt_b + h_m

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)
        e_all = torch.cat([e_gt, e_mtc], dim=1)

        logits = torch.einsum('bd,bnd->bn', h, e_all)
        ranks = logits[:, 1:].gt(logits[:, 0:1]).sum(dim=1).add(1)

        return ranks, mask_gt_a, mask_gt_b
