python ./src/main.py \
 --dataset dbp15k_zh_en_0_3 \
 --task entity-alignment \
 --hidden_size 256 \
 --num_hidden_layers 12 \
 --num_attention_heads 4 \
 --input_dropout_prob 0.5 \
 --attention_dropout_prob 0.1 \
 --hidden_dropout_prob 0.3 \
 --residual_dropout_prob 0.1 \
 --initializer_range 0.02 \
 --intermediate_size 2048 \
 --residual_w 0.5 \
 --epoch 500 \
 --min_epochs 200 \
 --learning_rate 5e-4 \
 --batch_size 2048 \
 --eval_batch_size 4096 \
 --early_stop_max_times 5 \
 --soft_label 0.25 \
 --eval_freq 5 \
 --start_eval 0 \
 --swa_pre_num 5 \
 --do_train true \
 --do_test true \
 --use_gelu false \
 --addition_loss_w 0.1 \
 --relation_combine_dropout_prob 0.2