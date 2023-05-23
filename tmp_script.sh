#AFHQ
sh_file_name="tmp_script.sh"
config="afhq.yml"
save_dir="./results/afhq"   # output directory
content_dir="./test_images/afhq/contents"
style_dir="./test_images/afhq/styles"
h_gamma=0.3             # Slerp ratio
t_boost=200             # 0 for out-of-domain style transfer.
n_gen_step=1000
n_inv_step=50
omega=0.0

CUDA_VISIBLE_DEVICES=0 python main.py --diff_style                       \
    --content_dir $content_dir                          \
    --style_dir $style_dir                              \
    --save_dir $save_dir                                \
    --config $config                                    \
    --n_gen_step $n_gen_step                            \
    --n_inv_step $n_inv_step                            \
    --n_test_step 1000                                  \
    --hs_coeff $h_gamma                                 \
    --t_noise $t_boost                                  \
    --sh_file_name $sh_file_name                        \
    --omega $omega                                      \
