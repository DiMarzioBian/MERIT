import torch.nn as nn


def init_weights(model: nn.Module,
                 ) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m.data, mean=0.0, std=0.02)

        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02, a=-0.04, b=0.04)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)
