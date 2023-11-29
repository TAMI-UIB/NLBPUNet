# A simple nonlocal back-projection unfolded network for pansharpening
Migars 2024 Fusion Net

## Abstract
Pansharpening is the fusion process that combines the geometry of a high-resolution panchromatic image with the
spectral information encoded in a low-resolution multispectral image. In this paper, we introduce a back-projection method
to minimize the reconstruction error between the target image and the output produced by the Brovey pansharpening model.
We replace the back-projection kernel with a residual network architecture that incorporates a nonlocal module,
exploiting self-similarity in satellite images and built upon the multi-head attention mechanism.
Experimental validation on the Pelican dataset showcases that our method achieves state-of-the-art results.

## Structure
![diagram.png](doc%2Fimg%2Fdiagram.png)
## Results

|          | ERGAS :arrow_down: | PSNR :arrow_up: | SAM :arrow_down: | SSIM :arrow_up: |
|----------|--------------------|-----------------|------------------|-----------------|
| Bicubic  | 64.97              | 34.35           | 7.82             | 0.9031          |
| Brovey   | 34.22              | 39.74           | 4.07             | 0.9816          |
| GSA      | 32.30              | 40.60           | 3.98             | 0.9789          |
| CNMF     | 31.75              | 40.12           | 3.98             | 0.9790          |
| GPPNN    | 17.90              | 45.57           | _2.17_           | 0.9920          |
| MDCUN    | _16.18_            | _46.26_         | 3.05             | _0.9924_        |
| NLRNet   | 16.90              | 45.81           | 2.03             | 0.9922          |
| **Ours** | **15.67**          | **46.54**       | **1.90**         | **0.9935**      |

## Usage example
Set classes in `config.py` for default values. Pass params to the `main.py` as string or key=value see example
```bash
python train --model GPPNN --model-params hyper_channels=4 --dataset-params scaling=4
```
To test new datasets (see StMichel as example):
1. Add <new class>.py to dataset directory
2. Add new class to `src/dataset/__init__.py`'s dict_dataset
3. Inputs and outputs should be dicts


To test new model (see NLBPUN as example):
1. Add <new class>.py to model directory
2. Add new class to `src/model/__init__.py`'s dict_model
3. Inputs and outputs should be dicts


Add an `.env` file in the project's root path with the following env-variables
```ini
DATASET_PATH="" # path to find all de datasets
SAVE_PATH="" # path to save tensorboard data
EVAL_FREQ=100  # % frequency to print to tensorboard
```
## SOTA
In this project we have compared our code with [MDCUN](https://github.com/yggame/MDCUN),
[NLRNet](https://github.com/Ding-Liu/NLRN) and [GPPNN](https://github.com/shuangxu96/GPPNN). 
And we get the classic methods from [Classic Methods](https://github.com/codegaj/py_pansharpening)
The original authors keep all the rights of the compared codes 
## Bibtex
