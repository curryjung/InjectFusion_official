#!/bin/bash

sh_file_name="script_diffstyle.sh"
gpu=3

config="celeba.yml"     # Other option: afhq.yml celeba.yml metfaces.yml ffhq.yml lsun_bedroom.yml ...
save_dir="./test/style_literature_gamma3"   # output directory
content_dir="./test_images/celeba/contents2"
style_dir="./test_images/style_literature"
h_gammas="0.3"
# dt_lambda=0.9985      # 1.0 for out-of-domain style transfer.
# t_boost=200           # 0 for out-of-domain style transfer.
n_gen_step=50
n_inv_step=50
# n_gen_step=1000
# n_inv_step=1000
# omega=0.0

for h_gamma in $h_gammas
do

CUDA_VISIBLE_DEVICES=$gpu python main.py --diff_style                       \
                        --content_dir $content_dir                          \
                        --style_dir $style_dir                              \
                        --save_dir $save_dir                                \
                        --config $config                                    \
                        --n_gen_step $n_gen_step                            \
                        --n_inv_step $n_inv_step                            \
                        --n_test_step 1000                                  \
                        --hs_coeff $h_gamma                                 \
                        --sh_file_name $sh_file_name                        \
                        # --user_defined_t_edit 500                           \
                        # --omega $omega \
                        # --use_mask

done
