from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

import utils
from utils.metrics import calculate_psnr_torch, calculate_ssim_torch


__all__ = [
    "data_transform",
    "inverse_data_transform",
    "DiffusiveRestoration",
]


def data_transform(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x - 1.0


def inverse_data_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion: Any, args: Any, config: Any) -> None:
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            raise FileNotFoundError("Pre-trained diffusion model path is missing.")

    def restore(self, val_loader: Any, validation: str = "snow", r: int | None = None) -> Dict[str, float]:
        image_folder = Path(self.args.image_folder) / self.config.data.dataset / validation
        image_folder.mkdir(parents=True, exist_ok=True)

        cumulative_psnr = 0.0
        cumulative_ssim = 0.0
        image_count = 0

        with torch.inference_mode():
            for _, (x, image_id) in enumerate(val_loader):
                print(f"starting processing from image {image_id}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device, non_blocking=True)
                x_gt = x[:, 3:, :, :].to(self.diffusion.device, non_blocking=True)

                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)

                #############################################
                #x_output = x_output.float()
                #x_gt = x_gt.to(device=x_output.device, dtype=x_output.dtype, non_blocking=True)

                batch_psnr = calculate_psnr_torch(x_output, x_gt)
                batch_ssim = calculate_ssim_torch(x_output, x_gt)

                batch_size = x_output.size(0)
                cumulative_psnr += batch_psnr.item() * batch_size
                cumulative_ssim += batch_ssim.item() * batch_size
                image_count += batch_size

                print(f"image {image_id} -> PSNR: {batch_psnr.item():.4f}, SSIM: {batch_ssim.item():.4f}")
                utils.logging.save_image(x_output, str(image_folder / f"{image_id}_output.png"))

        if image_count == 0:
            raise RuntimeError("Validation loader returned no images for restoration.")

        metrics = {
            "psnr": cumulative_psnr / image_count,
            "ssim": cumulative_ssim / image_count,
            "num_images": float(image_count),
        }
        print(f"{validation} set average -> PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}")
        return metrics

    def diffusive_restoration(self, x_cond: torch.Tensor, r: int | None = None) -> torch.Tensor:
        patch_size = int(self.config.data.image_size)
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=patch_size, r=r)
        corners = [(h, w) for h in h_list for w in w_list]
        x = torch.randn_like(x_cond, device=self.diffusion.device)
        return self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=patch_size)

    @staticmethod
    def _cover_positions(limit: int, output_size: int, stride: int) -> List[int]:
        if output_size > limit:
            raise ValueError(
                f"Patch size {output_size} cannot exceed image extent {limit}. "
                "Reduce config.data.image_size or resize inputs before inference."
            )
        if output_size == limit:
            return [0]

        positions = list(range(0, limit - output_size + 1, stride))
        last = limit - output_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def overlapping_grid_indices(
        self,
        x_cond: torch.Tensor,
        output_size: int,
        r: int | None = None,
    ) -> Tuple[List[int], List[int]]:
        _, _, h, w = x_cond.shape
        stride = 16 if r is None else int(r)
        if stride <= 0:
            raise ValueError(f"Stride must be positive, but got {stride}.")

        h_list = self._cover_positions(h, output_size, stride)
        w_list = self._cover_positions(w, output_size, stride)
        return h_list, w_list