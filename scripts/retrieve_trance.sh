#!/bin/bash

DATA_PATH=../TRANCE_dataset/trance_regen_100
python -m torch.distributed.run --nproc_per_node=1 --master_port=6667 main_task_retrieval.py \
--do_train \
--num_thread_reader=4 \
--epochs=20 \
--batch_size=8 \
--n_display=10 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/image \
--output_dir ckpts/ckpt_trance_retrieval \
--lr 1e-4 \
--max_words 64 \
--batch_size_val 8 \
--datatype trance \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32
