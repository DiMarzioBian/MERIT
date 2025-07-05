from os.path import join
from typing import Tuple
import argparse
import numpy as np
import pickle
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


def trim_seq(seq: np.array,
             len_trim: np.array,
             ) -> np.array:
    """ pad sequences to required length """
    return np.concatenate((np.zeros(max(0, len_trim - len(seq)), dtype=np.int32), seq))[-len_trim:]


def get_last_idx(seq: np.array,
                 len_trim: np.array,
                 ) -> np.array:
    """ make ground truths for domain-specific sequences """
    i = 0
    for i in range(-1, -min(len_trim, len(seq)) - 1, -1):
        if seq[i] != 0:
            break
    return np.array([i])


def get_gt_spe(seq: np.array,
               seq_raw: np.array,
               n_item_a: int,
               ) -> np.array:
    """ make domain-specific ground truths """
    seq_a = np.where(seq <= n_item_a, seq, 0)
    seq_b = np.where(seq > n_item_a, seq, 0)

    gt_raw_a = np.append(seq_raw[seq_raw <= n_item_a][1:], 0)
    gt_raw_b = np.append(seq_raw[seq_raw > n_item_a][1:], 0)

    gt_ab = np.array([], dtype=np.int32)

    for i, (i_a, i_b) in enumerate(zip(seq_a, seq_b)):
        if i_a > 0:
            gt_ab = np.append(gt_ab, gt_raw_a[0])
            gt_raw_a = gt_raw_a[1:]

        elif i_b > 0:
            gt_ab = np.append(gt_ab, gt_raw_b[0])
            gt_raw_b = gt_raw_b[1:]

        else:
            return AssertionError

    return gt_ab


def process_train(seq_raw: np.array,
                  n_item_a: int,
                  len_trim: int,
                  ) -> Tuple[np.array, np.array, np.array, list]:
    """ process training sequences """
    seq, gt_m = seq_raw[:-1], seq_raw[1:]

    gt_ab = get_gt_spe(seq, seq_raw, n_item_a)

    return *(trim_seq(x, len_trim) for x in (seq, gt_m, gt_ab)), seq_raw


def process_evaluate(seq_raw: np.array,
                     n_item_a: int,
                     len_trim: int,
                     ) -> Tuple[np.array, np.array, np.array, np.array, np.array, list]:
    """ process evaluation sequences """
    seq, gt = seq_raw[:-1], seq_raw[-1:]

    idx_last_a = get_last_idx(np.where(seq <= n_item_a, seq, 0), len_trim)

    idx_last_b = get_last_idx(np.where(seq > n_item_a, seq, 0), len_trim)

    return trim_seq(seq, len_trim), idx_last_a, idx_last_b, gt, seq_raw


def get_dataset(args: argparse,
                ) -> Tuple[Dataset, Dataset, Dataset]:
    """ get datasets """
    if args.raw:
        print('Reading raw data...')
        with open(join(args.path_data, f'map_item_{args.len_max}.txt'), 'r') as f:
            map_i = json.load(f)
            list_dm = np.array(list(map_i.values()))[:, 1]
            args.n_item_a = n_item_a = np.sum(list_dm == 0)
            args.n_item_b = n_item_b = np.sum(list_dm == 1)

        data_seq = []
        with open(join(args.path_data, args.f_raw), 'r', encoding='utf-8') as f:
            for line in f:
                seq = []
                line = line.strip().split(' ')
                for ui in line[1:][-args.len_max:]:
                    seq.append(int(ui.split('|')[0]))

                data_seq.append(np.array(seq))

        print('Serializing data...')
        data_tr = []
        data_val = []
        data_te = []
        len_trim = args.len_trim

        for seq in tqdm(data_seq, desc='processing', leave=False):
            data_tr.append(process_train(seq[:-2], n_item_a, len_trim))
            data_val.append(process_evaluate(seq[:-1], n_item_a, len_trim))
            data_te.append(process_evaluate(seq, n_item_a, len_trim))

        print('Saving serialized seqs...')
        with open(args.f_data, 'wb') as f:
            pickle.dump((data_tr, data_val, data_te, args.n_item_a, args.n_item_b), f)

    else:
        print('Loading serialized seqs...')
        with open(args.f_data, 'rb') as f:
            (data_tr, data_val, data_te, args.n_item_a, args.n_item_b) = pickle.load(f)

    args.n_item = args.n_item_a + args.n_item_b
    args.n_user = len(data_tr)
    assert len(data_tr) == len(data_val) == len(data_te)

    return TrainDataset(args, data_tr), EvalDataset(args, data_val), EvalDataset(args, data_te)


class TrainDataset(Dataset):
    """ training dataset """

    def __init__(self,
                 args: argparse,
                 data: list[list],
                 ) -> None:
        self.len_trim = args.len_trim
        self.n_neg = args.n_neg
        self.n_neg_x2 = args.n_neg * 2
        self.n_item_a = args.n_item_a

        self.data = data
        self.length = len(self.data)

        self.idx_all_a = np.arange(1, args.n_item_a + 1)
        self.idx_all_b = np.arange(args.n_item_a, args.n_item + 1)

    def get_m_neg(self,
                  gt: np.array,
                  cand_a: np.array,
                  cand_b: np.array,
                  ) -> np.array:
        gt_neg = np.zeros((self.len_trim, self.n_neg_x2))

        for i, x in enumerate(gt):
            if x != 0:
                gt_neg[i] = np.concatenate((np.random.default_rng().choice(cand_a, size=self.n_neg, replace=False),
                                            np.random.default_rng().choice(cand_b, size=self.n_neg, replace=False)))

        return gt_neg

    def get_ab_neg(self,
                   gt: np.array,
                   cand_a: np.array,
                   cand_b: np.array,
                   ) -> np.array:
        gt_neg = np.zeros((self.len_trim, self.n_neg))

        for i, x in enumerate(gt):
            if 0 < x <= self.n_item_a:
                gt_neg[i] = np.random.default_rng().choice(cand_a, size=self.n_neg, replace=False)

            elif x > self.n_item_a:
                gt_neg[i] = np.random.default_rng().choice(cand_b, size=self.n_neg, replace=False)

        return gt_neg

    def __len__(self):
        return self.length

    def __getitem__(self,
                    index: int,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_m, gt_m, gt_ab, seq_raw = self.data[index]

        cand_a = self.idx_all_a[~np.isin(self.idx_all_a, seq_raw[seq_raw <= self.n_item_a], assume_unique=True)]
        cand_b = self.idx_all_b[~np.isin(self.idx_all_b, seq_raw[seq_raw > self.n_item_a], assume_unique=True)]

        gt_neg_m = self.get_m_neg(gt_m, cand_a, cand_b)
        gt_neg_ab = self.get_ab_neg(gt_ab, cand_a, cand_b)

        return tuple(map(lambda x: torch.LongTensor(x), (seq_m, gt_m, gt_ab, gt_neg_m, gt_neg_ab)))


class EvalDataset(Dataset):
    """ evaluation dataset """

    def __init__(self,
                 args: argparse,
                 data: list[list],
                 ) -> None:
        self.len_trim = args.len_trim
        self.n_item_a = args.n_item_a
        self.n_mtc = args.n_mtc
        self.n_rand = args.n_mtc + args.len_trim

        self.data = data
        self.length = len(self.data)

        self.idx_all_a = np.arange(1, args.n_item_a + 1)
        self.idx_all_b = np.arange(args.n_item_a, args.n_item + 1)

    def get_mtc(self,
                gt: np.array,
                seq_raw: list,
                ) -> np.array:
        if gt <= self.n_item_a:
            gt_mtc = np.random.default_rng().choice(
                self.idx_all_a[~np.isin(self.idx_all_a, seq_raw[seq_raw <= self.n_item_a], assume_unique=True)],
                size=self.n_mtc, replace=False)

        else:
            gt_mtc = np.random.default_rng().choice(
                self.idx_all_b[~np.isin(self.idx_all_b, seq_raw[seq_raw > self.n_item_a], assume_unique=True)],
                size=self.n_mtc, replace=False)

        return gt_mtc

    def __len__(self):
        return self.length

    def __getitem__(self,
                    index: int,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_m, idx_last_a, idx_last_b, gt, seq_raw = self.data[index]

        gt_mtc = self.get_mtc(gt, seq_raw)

        return tuple(map(lambda x: torch.LongTensor(x), (seq_m, idx_last_a, idx_last_b, gt, gt_mtc)))


def get_dataloader(args: argparse,
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, valid_set, test_set = get_dataset(args)
    trainloader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.n_worker, pin_memory=True)
    valloader = DataLoader(valid_set, batch_size=args.bse, shuffle=False, num_workers=args.n_worker, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=args.bse, shuffle=False, num_workers=args.n_worker, pin_memory=True)
    return trainloader, valloader, testloader
