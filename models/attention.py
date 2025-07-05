import argparse

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,
                 d_embed: int,
                 n_head: int,
                 len_trim: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm = nn.LayerNorm(d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((len_trim,
                                                    len_trim), True), diagonal=1),
                             persistent=False)

    def forward(self,
                h: torch.Tensor,
                mask: torch.Tensor,
                ) -> torch.Tensor:
        h = self.norm(h + self.dropout(self.mha(h, h, h,
                                                attn_mask=self.mask_causal,
                                                is_causal=True,
                                                need_weights=False)[0])) * mask
        return h


class CrossAttention(nn.Module):
    def __init__(self,
                 d_embed: int,
                 n_head: int,
                 len_trim: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm = nn.LayerNorm(d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((len_trim,
                                                    len_trim), True), diagonal=1),
                             persistent=False)

    def forward(self,
                h_q: torch.Tensor,
                h_kv: torch.Tensor,
                mask: torch.Tensor,
                ) -> torch.Tensor:
        h_q = self.norm(h_q + self.dropout(self.mha(h_q, h_kv, h_kv,
                                                    attn_mask=self.mask_causal,
                                                    is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_q


class CrossAttention2(nn.Module):
    """ 2-kv-input CA """
    def __init__(self,
                 d_embed: int,
                 n_head: int,
                 len_trim: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm = nn.LayerNorm(d_embed)

        mask_causal = torch.triu(torch.full((len_trim,
                                             len_trim), True), diagonal=1)
        mask_causal = torch.concat((mask_causal, mask_causal, mask_causal), dim=-1)
        self.register_buffer('mask_causal', mask_causal, persistent=False)

    def forward(self,
                h_q: torch.Tensor,
                h_kv1: torch.Tensor,
                h_kv2: torch.Tensor,
                mask: torch.Tensor,
                ) -> torch.Tensor:
        h_kv = torch.concat((h_kv1, h_kv2), dim=1)
        h_q = self.norm(h_q + self.dropout(self.mha(h_q, h_kv, h_kv,
                                                    attn_mask=self.mask_causal,
                                                    is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_q
