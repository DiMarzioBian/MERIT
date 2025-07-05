# MERIT-pytorch

This is the ***official*** Pytorch implementation of paper "Multi-Domain Enhancement via Residual Interwoven Transfer in Cross-Domain Sequential Recommendation" accepted by ACM Multimedia 2025.

## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /utils/preprocess.py.

## 2. Usage
Please check demo.sh on running on different datasets.


## 3. File Tree
```
MERIT/
├── data/
│   ├── abe/
│   │   ├── abe_50_preprocessed.txt
│   │   ├── abe_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── afk/
│   │   ├── afk_50_preprocessed.txt
│   │   ├── afk_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── amb/
│   │   ├── amb_50_preprocessed.txt
│   │   ├── amb_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── mapper_raw_file.py
│   └── prepare_amazon_data.py
├── models/
│   ├── data/
│   │   ├── dataloader.py
│   │   └── evaluation.py
│   ├── utils/
│   │   ├── initialization.py
│   │   └── position.py
│   ├── attention.py
│   ├── ffn.py
│   └── MERIT.py
├── demo.sh
├── main.py
├── noter.py
├── README.md
├── requirements.txt
└── trainer.py
```
