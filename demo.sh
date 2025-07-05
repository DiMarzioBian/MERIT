#!/bin/bash

# afk
python main.py --data afk --n_worker 28 --bs 256 --lr 1e-4 --l2 5e0 --lr_g 0.4 --v 0 --seed 3407 --cuda 0

# abe
python main.py --data abe --n_worker 28 --bs 256 --lr 1e-4 --l2 5e0 --lr_g 0.8 --v 0 --seed 3407 --cuda 0

# amb
python main.py --data amb --n_worker 28 --bs 256 --lr 1e-4 --l2 5e0 --lr_g 0.1 --v 0 --seed 3407 --cuda 0