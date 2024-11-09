import os
from typing import Literal

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import dsine.utils.utils as utils
from dsine.utils.projection import intrins_from_fov, intrins_from_txt
from dsine.projects.dsine.config import make_parser

# normalize
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DSINE:
    def __init__(self, model, device: str, intrinsics_path: str | None = None):
        self.model = model
        self.device = device
        self.intrinsics_path = intrinsics_path

        if intrinsics_path is not None:
            assert os.path.exists(
                intrinsics_path
            ), f"Intrinsics file not found: {intrinsics_path}"

            # File should contain fx, fy, cx, cy
            self.intrinsics = intrins_from_txt(
                intrinsics_path, device=device
            ).unsqueeze(0)
        else:
            self.intrinsics = None

    @staticmethod
    def load_model(
        args_path: str,
        checkpoint_path: str,
        device: str,
    ) -> "DSINE":
        args = make_parser().parse_args(["@" + args_path])
        architecture = args.NNET_architecture

        if architecture == "v00":
            from dsine.models.dsine.v00 import DSINE_v00 as DSINE_
        elif architecture == "v01":
            from dsine.models.dsine.v01 import DSINE_v01 as DSINE_
        elif architecture == "v02":
            from dsine.models.dsine.v02 import DSINE_v02 as DSINE_
        elif architecture == "v02_kappa":
            from dsine.models.dsine.v02_kappa import DSINE_v02_kappa as DSINE_

        model = DSINE_(args)
        model = utils.load_checkpoint(checkpoint_path, model)
        model.eval()

        return DSINE(model, device)

    @torch.no_grad()
    def predict(self, image_pil: Image.Image) -> np.ndarray:
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # ↓↓↓↓
        # NOTE: forward pass

        # pad input
        _, _, orig_H, orig_W = image.shape
        lrtb = utils.get_padding(orig_H, orig_W)
        image = F.pad(image, lrtb, mode="constant", value=0.0)
        image = normalize(image)

        # Get intrinsics.
        if self.intrinsics is None:
            # If intrinsics are not given, we just assume that the principal point is at the center,
            # and that the field-of-view is 60 degrees (feel free to modify this assumption)..
            intrinsics = intrins_from_fov(
                new_fov=60.0,
                H=orig_H,
                W=orig_W,
                device=self.device,
            ).unsqueeze(0)
        else:
            intrinsics = self.intrinsics

        intrinsics[:, 0, 2] += lrtb[0]
        intrinsics[:, 1, 2] += lrtb[2]

        norm_out = self.model(image, intrins=intrinsics, mode="test")[-1]
        norm_out = norm_out[
            :, :, lrtb[2] : lrtb[2] + orig_H, lrtb[0] : lrtb[0] + orig_W
        ]
        pred_norm = norm_out[:, :3, :, :]
        # ↑↑↑↑

        return pred_norm
