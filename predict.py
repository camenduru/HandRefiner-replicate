from __future__ import absolute_import, division, print_function
import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/HandRefiner')
sys.path.append('/content/HandRefiner/MeshGraphormer')
os.chdir('/content/HandRefiner')

from config import handrefiner_root

def load():
    paths = [handrefiner_root, os.path.join(handrefiner_root, 'MeshGraphormer'), os.path.join(handrefiner_root, 'preprocessor')]
    for p in paths:
        sys.path.insert(0, p)

load()

import argparse
import json
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

import cv2
import einops
import numpy as np
import torch
import random
from preprocessor.meshgraphormer import MeshGraphormerMediapipe
import ast

transform = transforms.Compose([           
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

def inference(image_path, prompt, seed, model):
    meshgraphormer = MeshGraphormerMediapipe()
    image = np.array(Image.open(image_path))
    raw_image = image
    H, W, C = raw_image.shape
    gen_count = 0

    file_name_raw = Path(image_path).stem
    depthmap, mask, info = meshgraphormer.get_depth(os.path.dirname(image_path), os.path.basename(image_path), 30)
    cv2.imwrite("img_depth.jpg", depthmap)
    cv2.imwrite("img_mask.jpg", mask)
    control = depthmap

    ddim_sampler = DDIMSampler(model)
    out_dir="/content/HandRefiner/output"
    num_samples = 1
    ddim_steps = 50
    guess_mode = False
    adaptive_control = False
    eval=False
    strength = 0.55
    scale = 9.0

    a_prompt = "realistic, best quality, extremely detailed"
    n_prompt = "fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"

    source = raw_image

    source = (source.astype(np.float32) / 127.5) - 1.0
    source = source.transpose([2, 0, 1])  # source is c h w

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    hint = control.astype(np.float32) / 255.0

    masked_image = source * (mask < 0.5)  # masked image is c h w

    mask = torch.stack([torch.tensor(mask) for _ in range(num_samples)], dim=0).to("cuda")
    mask = torch.nn.functional.interpolate(mask, size=(64, 64))

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    masked_image = torch.stack(
        [torch.tensor(masked_image) for _ in range(num_samples)], dim=0
    ).to("cuda")

    # this should be b,c,h,w
    masked_image = model.get_first_stage_encoding(model.encode_first_stage(masked_image))

    x = torch.stack([torch.tensor(source) for _ in range(num_samples)], dim=0).to("cuda")
    z = model.get_first_stage_encoding(model.encode_first_stage(x))

    cats = torch.cat([mask, masked_image], dim=1)

    hint = hint[
        None,
    ].repeat(3, axis=0)

    hint = torch.stack([torch.tensor(hint) for _ in range(num_samples)], dim=0).to("cuda")

    cond = {
        "c_concat": [cats],
        "c_control": [hint],
        "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
    }
    un_cond = {
        "c_concat": [cats],
        "c_control": [hint],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
    }


    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    if not adaptive_control:
        seed_everything(seed)
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            x0=z,
            mask=mask
        )
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        # print(x_samples.shape)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        if eval: # currently only works for batch size of 1 
            assert num_samples == 1, "MPJPE evaluation currently only works for batch size of 1"
            mpjpe = meshgraphormer.eval_mpjpe(x_samples[0], info)
            print(mpjpe)
        for i in range(num_samples):
            cv2.imwrite('/content/HandRefiner/output/output_image.jpg', cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR))
            gen_count += 1
            return '/content/HandRefiner/output/output_image.jpg'
    else:
        assert num_samples == 1, "Adaptive thresholding currently only works for batch size of 1"
        strengths = [1.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ref_mpjpe = None
        chosen_strength = None
        final_mpjpe = None
        chosen_sample = None
        count = 0
        for strength in strengths:
            seed_everything(seed)
            model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)]
                if guess_mode
                else ([strength] * 13)
            )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                x0=z,
                mask=mask
            )
            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)

            x_samples = (
                (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .clip(0, 255)
                .astype(np.uint8)
            )
            mpjpe = meshgraphormer.eval_mpjpe(x_samples[0], info)
            if count == 0:
                ref_mpjpe = mpjpe
                chosen_sample = x_samples[0]
            elif mpjpe < ref_mpjpe * 1.15:
                chosen_strength = strength
                final_mpjpe = mpjpe
                chosen_sample = x_samples[0]
                break
            elif strength == 0.9:
                final_mpjpe = ref_mpjpe
                chosen_strength = 1.0
            count += 1
        
        cv2.imwrite('/content/HandRefiner/output/output_image.jpg', cv2.cvtColor(x_samples[0], cv2.COLOR_RGB2BGR))
        gen_count += 1
        return '/content/HandRefiner/output/output_image.jpg'

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = create_model("control_depth_inpaint.yaml").cpu()
        self.model.load_state_dict(load_state_dict('/content/HandRefiner/models/inpaint_depth_control.ckpt', location='cuda'), strict=False)
        self.model = self.model.to("cuda")
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        prompt: str = Input(default="a person facing the camera, making a hand gesture, indoor"),
        seed: int = Input(default=1),
    ) -> Path:
        output_image = inference(input_image, prompt, seed, self.model)
        return Path(output_image)
