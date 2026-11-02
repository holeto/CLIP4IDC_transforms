#!/bin/bash
# Stage 2 with the grammar target: the image-pair embedding is the z of the GVAE decoder, initialised from
# stage 1 (scripts/retrieve_trance_grammar.sh). Scored by exact match of the decoded derivation (no Java needed).

DATA_PATH=${DATA_PATH:-../TRANCE_dataset/trance_gvae_g4d}
python -m torch.distributed.run --nproc_per_node=1 --master_port=5566 main_task_caption.py \
--do_train \
--num_thread_reader=4 \
--epochs=20 \
--batch_size=4 \
--n_display=1 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/image \
--output_dir ckpts/ckpt_trance_grammar_caption \
--lr 1e-4 \
--batch_size_val 4 \
--datatype trance \
--target grammar \
--grammar_mask init_desc \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/ckpt_trance_grammar_retrieval/pytorch_model.bin.19 \
--seed 42
