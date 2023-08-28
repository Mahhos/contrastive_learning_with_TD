#!/bin/bash
export dataset=$1 && export seed=$2 && export processed_data_dir="processed_data" && cd src && python mahshid_back_translation.py --input="../processed_data/${dataset}/seed-${seed}/train/data" --output="../processed_data/${dataset}/seed-${seed}/train/back_translation"
python mahshid_get_synonym.py --input="../processed_data/${dataset}/seed-${seed}/train/data" --output="../processed_data/${dataset}/seed-${seed}/train/synonym"

