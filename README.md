
# Training-free Style Transfer Emerges from h-space in Diffusion models

[![arXiv](https://img.shields.io/badge/arXiv-2303.15403-red)](https://arxiv.org/abs/2303.15403) [![project_page](https://img.shields.io/badge/project_page-orange)](https://curryjung.github.io/DiffStyle/)


> **Training-free Style Transfer Emerges from h-space in Diffusion models**<br>
> [Jaeseok Jeong*](https://drive.google.com/file/d/14uHCJLoR1AFydqV_neGjl1H2rjN4HBdv/view), [Mingi Kwon*](https://drive.google.com/file/d/1d1TOCA20KmYnY8RvBvhFwku7QaaWIMZL/view?usp=share_link), [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor) *denotes equal contribution  <br>
> Arxiv preprint.
>**Abstract**: <br>
    Diffusion models (DMs) synthesize high-quality images in various domains. However, controlling their generative process is still hazy because the intermediate variables in the process are not rigorously studied. 
    Recently, StyleCLIP-like editing of DMs is found in the bottleneck of the U-Net, named $h$-space. 
    In this paper, we discover that DMs inherently have disentangled representations for content and style of
    the resulting images: $h$-space contains the content and the skip connections convey the style. 
    Furthermore, we introduce a principled way to inject content of one image to another considering progressive nature of the generative process. 
    Briefly, given the original generative process, 1) the feature of the source content should be gradually blended, 
    2) the blended feature should be normalized to preserve the distribution, 
    3) the change of skip connections due to content injection should be calibrated. 
    Then, the resulting image has the source content with the style of the original image just like image-to-image translation. 
    Interestingly, injecting contents to styles of unseen domains produces harmonization-like style transfer. 
    To the best of our knowledge, our method introduces the first training-free feed-forward style transfer only with 
    an unconditional pretrained frozen generative network.
 

## Description
This repo includes the official Pytorch implementation of **DiffStyle**, Training-free Style Transfer Emerges from h-space in Diffusion models.

- **DiffStyle** offers training-free style mixing and harmonization-like style transfer capabilities through content injection on *h-space* of diffusion models.

<!-- teaser image here -->
![teaser](./src/teaser.png)
- **DiffStyle** allows (a) style mixing by content injection within the trained domain, (b) local style mixing by injecting
masked content features, and (c) harmonization-like style transfer with out-of-domain style references. All results are pro-
duced by frozen pretrained diffusion models. Furthermore, flexibility of DiffStyle enables content injection into any style.


## Getting Started
We recommend running our code using NVIDIA GPU + CUDA, CuDNN.

### Pretrained Models for DiffStyle
To manipulate soure images, the pretrained Diffuson models are required.


| Image Type to Edit |Size| Pretrained Model | Dataset | Reference Repo. 
|---|---|---|---|---
| Human face |256×256| Diffusion (Auto) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [SDEdit](https://github.com/ermongroup/SDEdit)
| Human face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [FFHQ](https://arxiv.org/abs/1812.04948) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| Church |256×256| Diffusion (Auto) | [LSUN-Bedroom](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Bedroom |256×256| Diffusion (Auto) | [LSUN-Church](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Dog face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [AFHQ-Dog](https://arxiv.org/abs/1912.01865) | [ILVR](https://github.com/jychoi118/ilvr_adm)
| Painting face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [METFACES](https://arxiv.org/abs/2006.06676) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| ImageNet |256x256| [Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | [ImageNet](https://image-net.org/index.php) | [Guided Diffusion](https://github.com/openai/guided-diffusion)

- The pretrained Diffuson models on 256x256 images in [CelebA-HQ](https://arxiv.org/abs/1710.10196), [LSUN-Church](https://www.yf.io/p/lsun), and [LSUN-Bedroom](https://www.yf.io/p/lsun) are automatically downloaded in the code. (codes from [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP))
- In contrast, you need to download the models pretrained on other datasets in the table and put it in `./pretrained` directory. 
- You can manually revise the checkpoint paths and names in `./configs/paths_config.py` file.



### Datasets 
To precompute latents and find the direction of *h-space*, you need about 100+ images in the dataset. You can use both **sampled images** from the pretrained models or **real images** from the pretraining dataset. 

If you want to use **real images**, check the URLs :
- [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ), [AFHQ-Dog](https://github.com/clovaai/stargan-v2), [LSUN-Church](https://www.yf.io/p/lsun), [LSUN-Bedroom](https://www.yf.io/p/lsun), [ImageNet](https://image-net.org/index.php), [METFACES](https://github.com/NVlabs/metfaces-dataset), [FFHQ](https://github.com/NVlabs/ffhq-dataset)

You can simply modify `./configs/paths_config.py` for dataset path.


## DiffStyle

We provide some examples of inference script for DiffStyle. (`script_diffstyle.sh`)
- Determine `content_dir`, `style_dir`, `save_dir`.

```
#AFHQ

config="afhq.yml"
save_dir="./results/afhq"   # output directory
content_dir="./test_images/afhq/contents"
style_dir="./test_images/afhq/styles"
h_gamma=0.3             # Slerp ratio
t_boost=200             # 0 for out-of-domain style transfer.
n_gen_step=1000
n_inv_step=50
omega=0.0

python main.py --diff_style                       \
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

```


```
#CelebA_HQ style mixing with feature mask

config="celeba.yml"
save_dir="./results/masked_style_mixing"   # output directory
content_dir="./test_images/celeba/contents"
style_dir="./test_images/celeba/styles"
h_gamma=0.3             # Slerp ratio
dt_lambda=0.9985        # 1.0 for out-of-domain style transfer.
t_boost=200             # 0 for out-of-domain style transfer.
n_gen_step=1000
n_inv_step=50
omega=0.0

python main.py --diff_style                       \
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
                        --omega $omega                                      \
                        --use_mask                                          \

```


```
#Harmonization-like style mixing with artistic references

config="celeba.yml"   
save_dir="./results/style_literature"   # output directory
content_dir="./test_images/celeba/contents2"
style_dir="./test_images/style_literature"
h_gamma=0.4
n_gen_step=1000
n_inv_step=1000

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

```





## Acknowledge
Codes are based on [Asryp](https://github.com/kwonminki/Asyrp_official) and [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP).
