gpu=$1
ssn=$2
CUDA_VISIBLE_DEVICES=$gpu python trainer.py \
    --rootpath dataset \
    --overwrite 1 \
    --max_violation \
    --text_norm \
    --visual_norm \
    --collection msrvtt10k \
    --visual_feature resnext101-resnet152 \
    --space latent \
    --batch_size 128 \
    --style GT \
    --postfix "GT_${ssn}" \
    --support_set_number ${ssn}

CUDA_VISIBLE_DEVICES=$gpu python trainer.py \
    --rootpath dataset \
    --overwrite 1 \
    --max_violation \
    --text_norm \
    --visual_norm \
    --collection msrvtt10k \
    --visual_feature resnext101-resnet152 \
    --space latent \
    --batch_size 128 \
    --style distill_from_best_model \
    --student_model text+video \
    --distill_loss text+video \
    --distill_with_triplet \
    --support_set_number $ssn \
    --teacher_model GT \
    --postfix "student_support_set_${ssn}" \
    --resume "dataset/msrvtt10k/support-set/msrvtt10k/dual_encoding_latent_concate_full_dp_0.2_measure_cosine_jaccard/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_resnext101-resnet152_visual_rnn_size_512_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-1536_img_0-1536_tag_vocab_size_512/loss_func_mrl_margin_0.2_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/GT_${ssn}/model_best.pth.tar" \
    --with_detach \
    --distill_with_similarity \
    --similarity_type diag


