#!/bin/bash

cfd=`pwd`
cd ../../onmt_mt
# python3 preprocess.py --train_src $cfd/data_alleng/alleng_src_training.txt --train_tgt $cfd/data_alleng/alleng_tgt_training.txt --save_data $cfd/data_alleng/data --shard_size 200000 --src_vocab_size 2000000 -tgt_vocab_size 2000000

# python3 tools/embeddings_to_torch.py -emb_file_enc $cfd/wiki.multi.en.vec -emb_file_dec $cfd/wiki.multi.en.vec -dict_file $cfd/data/shufflm.vocab.pt -type word2vec -output_file $cfd/data/embeddings

# python3 feature_emb_to_torch.py -emb_file pretrained_lang_emb.vec -output_file $cfd/data/pretrained_feature -dict_file $cfd/data/shufflm.vocab.pt

python3 train.py --data $cfd/data_alleng/data --save_model $cfd/alleng_shufflm_2/model --word_vec_size 300 --feat_vec_size 300 --feat_merge sum --encoder_type brnn --enc_rnn_size 500 --dec_rnn_size 500 --optim sgd --learning_rate .1 --batch_size 16 --train_steps 400000 --valid_batch_size 32 --start_decay_steps 10000 --decay_steps 25000 --learning_rate_decay .85 --gpu_ranks 0 --log_file $cfd/alleng_shufflm_2/training_log.txt

# python3 translate.py --src $cfd/validation_src.txt --tgt $cfd/validation_src.txt --output $cfd/model_validation_out.txt --report_bleu --model $cfd/shufflm/model_step_100000.pt --batch_size 32 --gpu 0