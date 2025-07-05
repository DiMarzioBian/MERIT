from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self,
                 d_input: int,
                 n_expert: int,
                 ) -> None:
        super().__init__()
        self.fc = nn.Linear(d_input, n_expert, bias=False)

    def forward(self,
                h: torch.Tensor,
                ) -> torch.Tensor:
        h = F.softmax(self.fc(h), dim=-1)
        return h


class MLP(nn.Module):
    """ SwiGLU-variant """
    def __init__(self,
                 d_embed: int,
                 d_ffn: int = 680,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_ffn = d_ffn
        self.fc_1 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_embed, bias=False)

    def forward(self,
                h: torch.Tensor,
                ) -> torch.Tensor:
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class FeedForward(nn.Module):
    """ MLP with post-norm residual connection """
    def __init__(self,
                 d_embed: int,
                 dropout: float = 0.1,
                 d_ffn: int = 680,
                 ) -> None:
        super().__init__()
        self.mlp = MLP(d_embed, d_ffn)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm = nn.LayerNorm(d_embed)

    def forward(self,
                h: torch.Tensor,
                mask: torch.Tensor = None,
                ) -> torch.Tensor:
        h = self.norm(h + self.dropout(self.mlp(h)))
        if mask is not None:
            h = h * mask
        return h


class MoFFN(nn.Module):
    """ 3-output MoE-FFN """
    def __init__(self,
                 d_embed: int,
                 dropout: float = 0.1,
                 d_ffn: int = 680,
                 ) -> None:
        super().__init__()
        self.expert_s = MLP(d_embed, d_ffn)  # shared
        self.expert_1 = MLP(d_embed, d_ffn)
        self.expert_2 = MLP(d_embed, d_ffn)
        self.expert_3 = MLP(d_embed, d_ffn)

        self.gate_1 = Gate(d_embed, 2)
        self.gate_2 = Gate(d_embed, 2)
        self.gate_3 = Gate(d_embed, 2)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.norm_1 = nn.LayerNorm(d_embed)
        self.norm_2 = nn.LayerNorm(d_embed)
        self.norm_3 = nn.LayerNorm(d_embed)

    def forward(self,
                h: torch.Tensor,
                mask: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_s = self.expert_s(h).unsqueeze(-2)
        h_1 = self.expert_1(h).unsqueeze(-2)
        h_2 = self.expert_2(h).unsqueeze(-2)
        h_3 = self.expert_3(h).unsqueeze(-2)

        g_1 = self.gate_1(h).unsqueeze(-1)
        g_2 = self.gate_2(h).unsqueeze(-1)
        g_3 = self.gate_3(h).unsqueeze(-1)

        h_1 = (g_1 * torch.cat([h_s, h_1], dim=-2)).sum(-2)
        h_2 = (g_2 * torch.cat([h_s, h_2], dim=-2)).sum(-2)
        h_3 = (g_3 * torch.cat([h_s, h_3], dim=-2)).sum(-2)

        h_1 = self.norm_1(h + self.dropout(h_1)) * mask
        h_2 = self.norm_2(h + self.dropout(h_2)) * mask
        h_3 = self.norm_3(h + self.dropout(h_3)) * mask

        return h_1, h_2, h_3
