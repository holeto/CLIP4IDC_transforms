#!/bin/bash

export JAVA_HOME="your_jdk/"
export PATH=$JAVA_HOME/bin:$PATH

DATA_PATH=../TRANCE_dataset/trance_regen_100
# the references for scoring come from: python preprocess/trance_gt.py ${DATA_PATH}
python -m torch.distributed.run --nproc_per_node=1 --master_port=5565 main_task_caption.py \
--do_train \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=8 \
--n_display=10 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/image \
--gt_dir ${DATA_PATH} \
--output_dir ckpts/ckpt_trance_caption \
--lr 1e-4 \
--max_words 64 \
--batch_size_val 8 \
--datatype trance \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/ckpt_trance_retrieval/pytorch_model.bin.19 \
--seed 42
