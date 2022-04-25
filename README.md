# X-Mixup

Implementation of ICLR 2022 paper "[Enhancing Cross-lingual Transfer by Manifold Mixup](https://openreview.net/pdf?id=OjPmfr9GkVv)" (updating...).


## Structure
```text
.
├── data                              # XTREME data and translation data
│   ├── xnli
│   │   ├── XNLI-MT-1.0
│   │   ├── XNLI-1.0          
│   │   ├── translate-test
│   │   ├── translate-dev
│   │   ├── translate-train-en  # back-translation data
│   ├── pawsx
│   │   ├── en (each language)
│   │   │   ├── train.tsv
│   │   │   ├── dev_2k.tsv
│   │   │   ├── test_2k.tsv        
│   │   ├── translate-test
│   │   ├── translate-dev
│   │   ├── translate-train-en
│   ├── udpos/panx
│   │   ├── en (each language) 
│   │   │   ├── train.tsv (only en)
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   ├── translate-train-logits
│   │   ├── translate-test-token
│   │   ├── translate-dev-token
│   │   ├── translate-train-en-token
│   ├── squad
│   │   ├── train-v1.1.json
│   │   ├── translate-train
│   │   ├── translate-train-en
│   ├── mlqa
│   │   ├── dev
│   │   ├── test
│   │   ├── translate-test
│   ├── xquad
│   │   ├── test
│   │   ├── translate-test
│   ├── tydiqa
│   │   ├── dev
│   │   ├── translate-test
│   │   ├── translate-train
│   │   ├── translate-train-en         
├── scripts
├── xmixup
├── README.md
└── requirements.txt
```

## Environment
```bash
pip install -r requirements.txt
```

## Data
Prepare data before the training phrase: 
* Step 1: Download XTREME data from [XTREME repo](https://github.com/google-research/xtreme) (Note that we should keep the label of test set for evaluation).
* Step 2: Download other translation data from xxx (updating...).
* Step 3: Organize data following the Structure part.

## Training & Evaluation
```bash
bash ./scripts/train.sh [pretrained_model] [task_name] [data_dir] [output_dir]
```
where the options are described as follows:
- `[pretrained_model]`: `xlmr` or `mbert`
- `[task_name]`: `pawsx`, `xnli`, `udpos`, `panx`, `mlqa`, `xquad`, `tydiqa`
- `[data_dir]`: data directory
- `[output_dir]`: output directory

## Citation
Please consider citing our papers in your publications if this project helps your research
```
@inproceedings{yang2021enhancing,
  title={Enhancing Cross-lingual Transfer by Manifold Mixup},
  author={Yang, Huiyun and Chen, Huadong and Zhou, Hao and Li, Lei},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
