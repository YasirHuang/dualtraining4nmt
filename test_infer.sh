export CUDA_VISIBLE_DEVICES=1
/home/xhuang/anaconda2/envs/tensorflow-gpu/bin/python test_infer.py \
    --time_major=True \
    --pass_hidden_state=False \
	--attention=scaled_luong \
	--src=en \
	--tgt=de \
	--rootdir=./experiment/data \
	--vocab_prefix=./experiment/data/vocab.tok.lc.bpe \
	--train_prefix=./experiment/data/train.tok.lc.bpe \
	--dev_prefix=./experiment/data/val.tok.lc.bpe \
	--test_prefix=./experiment/data/test.tok.lc.bpe \
	--out_dir=experiment/model_20180329 \
	--epoch=25 \
	--steps_per_eval=100 \
	--num_units=1000 \
	--num_embedding=620 \
	--attention_architecture=standard \
	--beam_width=3 \
	--learning_rate=0.001 \
	--unit_type=gru \
	--dropout=0.8 \
	--bpe_delimiter=@@ \
	--optimizer=adam \
	--source_reverse=False\
	--batch_size=40