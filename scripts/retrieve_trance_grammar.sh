#!/bin/bash
# Stage 1 with the grammar target: GVAE encoder in place of CLIP's text tower (modules/modeling_grammar.py).
# The TRANCE grammar is imported from ../TVR_multi_sample unless TRANCE_GRAMMAR_ROOT is set.
# Needs data from the generator with the grammar constraints (2026-10-26 or later); --grammar_mask init_desc|program|cfg.

DATA_PATH=${DATA_PATH:-../TRANCE_dataset/trance_gvae_g4d}
python -m torch.distributed.run --nproc_per_node=1 --master_port=6668 main_task_retrieval.py \
--do_train \
--num_thread_reader=4 \
--epochs=20 \
--batch_size=4 \
--n_display=1 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/image \
--output_dir ckpts/ckpt_trance_grammar_retrieval \
--lr 1e-4 \
--batch_size_val 4 \
--datatype trance \
--target grammar \
--grammar_mask init_desc \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32
