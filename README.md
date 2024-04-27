# Text-to-Image Editing Ai Tool

## HD-Painter 
### This repository is the official implementation of [HD-Painter](https://arxiv.org/abs/2312.14091)
### [link](https://github.com/Picsart-AI-Research/HD-Painter/tree/main) to the main repo 





## explainatioin

The repository provided contains the code for HD-Painter, a program developed by Picsart AI Research. HD-Painter is designed for high-resolution image painting. The AI model used in this program is likely a deep neural network, possibly a variant of Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs), trained on a large dataset of high-resolution images to generate detailed and realistic paintings or image enhancements.

- Generative Adversarial Networks (GANs) : GANs consist of two networks, a generator and a discriminator, trained adversarially to generate realistic images. The generator creates images from random noise, while the discriminator learns to distinguish between real and generated images.

- Variational Autoencoders (VAEs) : VAEs learn a latent representation of images and generate new samples from this learned distribution. They consist of an encoder network that maps input images to a latent space and a decoder network that reconstructs images from this latent space.





## Setup 

python3.9 and above needed to install all the dependencies 

```
python3 -m venv venv
source ./venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
```

To perform metric evaluation, additionally install ```mmcv ``` by running:
```
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

## Usage

You can use the following script to perform inference on the given image+mask pair and prompt:
 
```
python hd_inpaint.py \
  --model-id ONE_OF[ds8_inp, sd2_inp, sd15_inp] \
  --method ONE_OF[baseline, painta, rasg, painta+rasg] \
  --image-path HR_IMAGE_PATH \
  --mask-path HR_IMAGE_MASK \
  --prompt PROMPT_TXT \
  --output-dir OUTPUT_DIRECTORY
```

`--model-id` specifies the baseline model for text-guided image inpainting. The following baseline models are supported by the script:
- `ds8_inp` - DreamShaper 8 Inpainting
- `sd2_inp` - Stable Diffusion 2.0 Inpainting
- `sd15_inp` - Stable Diffusion 1.5 Inpainting

If not specified `--model-id` defaults to `ds8_inp`.

` --method` specifies the inpainting method. The available options are as such:
- `baseline` - Run the underlying baseline model.
- `painta` - Use PAIntA block as introduced in the paper.
- `rasg` - Use RASG guidance mechanism as introduced in the paper.
- `painta+rasg` - Use both PAIntA and RASG mechanisms.
 
If not specified `--method` defaults to `painta+rasg`.

The script uses combination of positive and negative text prompts by default for visually more pleasing results.

The script will automatically download the necessary models during first run, please be patient. Output will be saved in the `--output-dir` directory. Please note that the provided script outputs an image which longer side is equal to 2048px while the aspect ratio is preserved. You can see more options and details in `hd_inpaint.py` script.

## Gradio Demo

From the project root folder, run this shell command:
```
python demo/app.py
```

# Results

The results of the [input image](https://github.com/Shashanksharma280201/Text-to-Image-editing-ai-tool-/tree/eec92b9bc4b98575741a1547cf3b47194081866c/Input%20image) are in the  [results](https://github.com/Shashanksharma280201/Text-to-Image-editing-ai-tool-/tree/acd66069092d100469bdb4571e2a7fda3fa91cb3/results) folder



