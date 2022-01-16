python ./src/main.py \
 --dataset wn18rr \
 --task link-prediction \
 --hidden_size 256 \
 --num_hidden_layers 8 \
 --num_attention_heads 2 \
 --input_dropout_prob 0.3 \
 --attention_dropout_prob 0.2 \
 --hidden_dropout_prob 0.1 \
 --residual_dropout_prob 0.2 \
 --initializer_range 0.02 \
 --intermediate_size 2048 \
 --residual_w 0.2 \
 --epoch 500 \
 --min_epochs 300 \
 --learning_rate 5e-4 \
 --batch_size 2048 \
 --eval_batch_size 4096 \
 --early_stop_max_times 2 \
 --soft_label 0.25 \
 --eval_freq 5 \
 --start_eval 0 \
 --swa_pre_num 20 \
 --do_train true \
 --do_test true \
 --use_gelu false \
 --addition_loss_w 0.1 \
 --relation_combine_dropout_prob 0.2 