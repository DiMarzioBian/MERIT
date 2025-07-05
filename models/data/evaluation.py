import numpy as np
import torch


def cal_norm_mask(mask: torch.Tensor,
                  ) -> torch.Tensor:
    """ calculate normalized mask """
    return mask * mask.sum(1).reciprocal().unsqueeze(-1)


def cal_mrr(ranks: list,
            ) -> float:
    """ calculate metrics mrr """
    return sum([1 / r for r in ranks]) / len(ranks)


def cal_metrics(ranks: list,
                ) -> list:
    """ calculate metrics hr@5, hr@5, ndcg@10 and mrr """
    N = len(ranks)
    hr5, hr10, ndcg10, mrr = 0., 0., 0., 0.

    for rank in ranks:
        mrr += 1 / rank
        if rank <= 10:
            hr10 += 1
            ndcg10 += 1 / np.log2(rank + 1)
            if rank <= 5:
                hr5 += 1
    return [x / N for x in (hr5, hr10, ndcg10, mrr)]
