# Stable Diffusion for Remote Sensing Image Generation

#### Author: Zhiqiang yuan @ AIR CAS,  [Send a Email](yuan_zhi_qiang@sina.cn)
#### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update
#### -------------------------------------------------------------------------------------

![./assets/MAIN.png](./assets/MAIN.gif)

A simple project for `text-to-image remote sensing image generation`,
and we will release the code of `using multiple text to control regions for super-large RS image generation` later.
Also welcome to see the project of [image-condition fake sample generation](https://github.com/xiaoyuan1996/Controllable-Fake-Sample-Generation-for-RS) in [TGRS, 2023](https://ieeexplore.ieee.org/abstract/document/10105619/).

## Environment configuration

This repository now supports a **diffusers-native** workflow for training, inference, and checkpoint conversion.

Install dependencies for the new workflow:

```bash
pip install -r requirements-diffusers.txt
```

Optional: use local editable Hugging Face diffusers source inside this repo:

```bash
git clone https://github.com/huggingface/diffusers.git external/diffusers
```

All new scripts automatically use `external/diffusers/src` when present.


## Pretrained weights

We used RS image-text dataset [RSITMD](https://github.com/xiaoyuan1996/AMFMN) as training data and fine-tuned stable diffusion for 10 epochs with 1 x A100 GPU.
When the batchsize is 4, the GPU memory consumption is about 40+ Gb during training, and about 20+ Gb during sampling.
The pretrain weights is realesed at [last-pruned.ckpt](https://drive.google.com/drive/folders/10vK3eNpIw7H3lxxZbB7NF2IZktGt95As?usp=sharing).

## Using (diffusers-native)

### 1) Convert checkpoint to diffusers format

General converter:

```bash
python scripts/convert_ckpt_to_diffusers_native.py \
  --checkpoint_path ./last-pruned.ckpt \
  --original_config_file ./configs/stable-diffusion/RSITMD.yaml \
  --output_path ./outputs/rsitmd-diffusers
```

RSITMD shortcut:

```bash
python scripts/convert_rsitmd_ckpt_to_diffusers.py \
  --checkpoint_path ./last-pruned.ckpt \
  --output_path ./outputs/rsitmd-diffusers
```

### 2) Inference with native diffusers

```bash
python scripts/diffusers_txt2img.py \
  --model_path ./outputs/rsitmd-diffusers \
  --prompt "Some boats drived in the sea" \
  --outdir outputs/RS \
  --height 512 \
  --width 512 \
  --num_images 4
```

### 3) Training with native diffusers + accelerate

Put RSITMD images under `data/RSITMD/images`, then run:

```bash
accelerate launch scripts/diffusers_train_rsitmd.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_json_path data/RSITMD/hf_train.json \
  --data_root data/RSITMD \
  --output_dir outputs/diffusers-rsitmd \
  --resolution 512 \
  --train_batch_size 4 \
  --num_train_epochs 10 \
  --learning_rate 1e-5
```

### Legacy entrypoints (deprecated)

- `python main.py ...`
- `python scripts/txt2img.py ...`

The legacy LDM pipeline is kept for compatibility only; prefer the diffusers-native scripts above.


## Examples
**Caption:** `Some boats drived in the sea.`
![./assets/shows1.png](./assets/shows1.png)

**Caption:** `A lot of cars parked in the airport.`
![./assets/shows2.png](./assets/shows2.png)

**Caption:** `A large number of vehicles are parked in the parking lot, next to the bare desert.`
![./assets/shows3.png](./assets/shows3.png)

**Caption:** `There is a church in a dark green forest with two table tennis courts next to it.`
![./assets/shows4.png](./assets/shows4.png)
