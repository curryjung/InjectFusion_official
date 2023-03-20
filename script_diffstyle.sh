#!/bin/bash

sh_file_name="script_diffstyle.sh"
gpu=0

config="celeba.yml"     # Other option: afhq.yml celeba.yml metfaces.yml ffhq.yml lsun_bedroom.yml ...
save_dir="./bin/1000_test_no_omega_3gamma"   # output directory
content_dir="./test_images/celeba/contents"
style_dir="./test_images/celeba/styles"
h_gamma=0.3
dt_lambda=0.9985      # 1.0 for out-of-domain style transfer.
t_boost=200           # 0 for out-of-domain style transfer.
# n_gen_step=50
# n_inv_step=50
n_gen_step=1000
n_inv_step=1000
omega=0.0

CUDA_VISIBLE_DEVICES=$gpu python main.py --diff_style                       \
                        --content_dir $content_dir                          \
                        --style_dir $style_dir                              \
                        --save_dir $save_dir                                \
                        --config $config                                    \
                        --n_gen_step $n_gen_step                            \
                        --n_inv_step $n_inv_step                            \
                        --n_test_step 1000                                  \
                        --dt_lambda $dt_lambda                              \
                        --hs_coeff $h_gamma                                 \
                        --t_noise $t_boost                                  \
                        --sh_file_name $sh_file_name                        \
                        --user_defined_t_edit 500                           \
                        --omega $omega \
                        # --use_mask
