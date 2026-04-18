from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import utils
from models.unet import DiffusionUNet
from utils.metrics import calculate_psnr_torch, calculate_ssim_torch


__all__ = [
    "data_transform",
    "inverse_data_transform",
    "EMAHelper",
    "get_beta_schedule",
    "noise_estimation_loss",
    "DenoisingDiffusion",
]


def data_transform(x: torch.Tensor) -> torch.Tensor:
    """Map image tensors from [0, 1] to [-1, 1]."""
    return 2.0 * x - 1.0


def inverse_data_transform(x: torch.Tensor) -> torch.Tensor:
    """Map model-space tensors from [-1, 1] back to [0, 1]."""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def unwrap_module(module: nn.Module) -> nn.Module:
    """Return the underlying module for DataParallel/DDP-wrapped models."""
    return module.module if hasattr(module, "module") else module


class EMAHelper:
    """Simple exponential moving average tracker for trainable parameters."""

    def __init__(self, mu: float = 0.9999) -> None:
        self.mu = float(mu)
        self.shadow: Dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module) -> None:
        module = unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> None:
        for name, tensor in self.shadow.items():
            self.shadow[name] = tensor.to(
                device=device,
                dtype=dtype if dtype is not None else tensor.dtype,
            )

    def update(self, module: nn.Module) -> None:
        module = unwrap_module(module)
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue

            shadow_param = self.shadow[name]
            if shadow_param.device != param.device or shadow_param.dtype != param.dtype:
                shadow_param = shadow_param.to(device=param.device, dtype=param.dtype)
                self.shadow[name] = shadow_param

            shadow_param.mul_(self.mu).add_(param.detach(), alpha=1.0 - self.mu)

    def ema(self, module: nn.Module) -> None:
        module = unwrap_module(module)
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            shadow_param = self.shadow[name]
            if shadow_param.device != param.device or shadow_param.dtype != param.dtype:
                shadow_param = shadow_param.to(device=param.device, dtype=param.dtype)
                self.shadow[name] = shadow_param

            param.data.copy_(shadow_param.data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        inner_module = unwrap_module(module)
        module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
        module_copy.load_state_dict(inner_module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}


def get_beta_schedule(
    beta_schedule: str,
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """Construct a beta schedule for the forward diffusion process."""

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (np.exp(-x) + 1.0)

    if beta_schedule == "quad":
        betas = np.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        ) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        xs = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(xs) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    if betas.shape != (num_diffusion_timesteps,):
        raise ValueError(f"Unexpected beta shape: {betas.shape}")
    return betas


def _resolve_amp_dtype(device: torch.device, amp_dtype: Optional[str]) -> torch.dtype:
    if device.type == "cuda":
        mapping = {
            None: torch.float16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
    else:
        mapping = {
            None: torch.bfloat16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }

    if amp_dtype not in mapping:
        raise ValueError(f"Unsupported amp dtype '{amp_dtype}' for device type '{device.type}'.")
    return mapping[amp_dtype]


def _create_grad_scaler(enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    try:
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def noise_estimation_loss(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    psnr_weight: float = 0.0,
    ssim_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    x0 is expected to be channel-concatenated as [condition, ground_truth].
    """
    a = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
    x_cond = x0[:, :3, :, :]
    x_gt = x0[:, 3:, :, :]
    x_t = x_gt * a.sqrt() + e * (1.0 - a).sqrt()

    model_input = torch.cat([x_cond, x_t], dim=1)
    output = model(model_input, t.float())

    noise_loss = F.mse_loss(output, e, reduction="mean")

    denom = a.sqrt().clamp(min=1e-8)
    x0_pred = (x_t - output * (1.0 - a).sqrt()) / denom
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

    x0_pred_img = inverse_data_transform(x0_pred)
    x0_gt_img = inverse_data_transform(x_gt)

    psnr_value = calculate_psnr_torch(x0_pred_img, x0_gt_img)
    ssim_value = calculate_ssim_torch(x0_pred_img, x0_gt_img)

    psnr_loss = 1.0 / psnr_value.clamp(min=1e-8)
    ssim_loss = 1.0 - ssim_value
    total_loss = noise_loss + psnr_weight * psnr_loss + ssim_weight * ssim_loss

    return {
        "total_loss": total_loss,
        "noise_loss": noise_loss.detach(),
        "psnr_loss": psnr_loss.detach(),
        "ssim_loss": ssim_loss.detach(),
        "psnr": psnr_value.detach(),
        "ssim": ssim_value.detach(),
    }


class DenoisingDiffusion:
    def __init__(self, args: Any, config: Any) -> None:
        self.args = args
        self.config = config
        self.device = config.device

        base_model = DiffusionUNet(config).to(self.device)

        use_compile = bool(getattr(config.model, "compile", False))
        use_data_parallel = bool(getattr(config.training, "use_data_parallel", True))
        use_multi_gpu = self.device.type == "cuda" and torch.cuda.device_count() > 1 and use_data_parallel

        if use_compile and hasattr(torch, "compile") and not use_multi_gpu:
            base_model = torch.compile(base_model)

        self.model: nn.Module = nn.DataParallel(base_model) if use_multi_gpu else base_model

        self.ema_helper = EMAHelper(mu=float(getattr(config.model, "ema_rate", 0.9999)))
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch = 0
        self.step = 0
        self.best_loss = float("inf")

        use_amp = bool(getattr(config.training, "use_amp", False)) and self.device.type in {"cuda", "cpu"}
        self.use_amp = use_amp
        self.amp_dtype = _resolve_amp_dtype(self.device, getattr(config.training, "amp_dtype", None))
        self.scaler = _create_grad_scaler(enabled=use_amp and self.device.type == "cuda")

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.num_timesteps = self.betas.shape[0]

        ckpt_dir = Path(self.config.data.data_dir) / "ckpts"
        self.checkpoint_saver = utils.logging.CheckpointSaver(
            save_dir=ckpt_dir,
            prefix=f"{self.config.data.dataset}_ddpm",
        )

        plot_every_n_events = int(getattr(self.config.training, "plot_freq", 1))
        trace_dir = Path(self.config.data.data_dir) / "training_logs"
        self.metric_tracker = utils.logging.LiveMetricTracker(
            log_dir=trace_dir,
            prefix=f"{self.config.data.dataset}_ddpm",
            plot_every_n_events=plot_every_n_events,
        )

    def load_ddm_ckpt(self, load_path: str, ema: bool = False) -> None:
        checkpoint = utils.logging.load_checkpoint(load_path, device="cpu")

        self.start_epoch = int(checkpoint["epoch"])
        self.step = int(checkpoint["step"])
        self.best_loss = float(checkpoint.get("best_loss", float("inf")))

        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(self.optimizer, self.device)

        self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        self.ema_helper.to(self.device)

        if ema:
            self.ema_helper.ema(self.model)

        if math.isfinite(self.best_loss):
            print(
                f"=> loaded checkpoint '{load_path}' "
                f"(epoch {self.start_epoch}, step {self.step}, best_loss {self.best_loss:.6f})"
            )
        else:
            print(f"=> loaded checkpoint '{load_path}' (epoch {self.start_epoch}, step {self.step})")

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    def _build_checkpoint_state(self, epoch: int, current_loss: float) -> Dict[str, Any]:
        return {
            "epoch": int(epoch + 1),
            "step": int(self.step),
            "current_loss": float(current_loss),
            "best_loss": float(self.best_loss),
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_helper": self.ema_helper.state_dict(),
        }

    def _save_snapshots(self, epoch: int, current_loss: float) -> None:
        old_best_loss = self.best_loss
        is_best = current_loss < old_best_loss

        if is_best:
            self.best_loss = current_loss

        state = self._build_checkpoint_state(epoch, current_loss)
        state["best_loss"] = float(self.best_loss)

        last_path = self.checkpoint_saver.save_last(state)
        best_updated, _, best_path = self.checkpoint_saver.save_best(
            state,
            current_loss=current_loss,
            best_loss=old_best_loss,
        )

        self.metric_tracker.log_snapshot(
            epoch=epoch,
            step=self.step,
            current_loss=current_loss,
            best_loss=self.best_loss,
            is_best=is_best,
        )

        if best_updated:
            print(
                f"last snapshot saved: {last_path} | "
                f"best snapshot updated: {best_path} | "
                f"best model loss: {self.best_loss:.6f}"
            )
        else:
            print(
                f"last snapshot saved: {last_path} | "
                f"best snapshot kept: {best_path} | "
                f"best model loss: {self.best_loss:.6f}"
            )

    def train(self, dataset_builder: Any) -> None:
        if self.device.type == "cuda":
            cudnn.benchmark = True

        train_loader, val_loader = dataset_builder.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        log_freq = int(getattr(self.config.training, "log_freq", 10))

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print(f"epoch: {epoch}")
            data_start = time.time()
            data_time = 0.0

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device, non_blocking=True)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])

                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    loss_dict = noise_estimation_loss(
                        self.model,
                        x,
                        t,
                        e,
                        self.alphas_cumprod,
                        psnr_weight=float(getattr(self.config.training, "psnr_weight", 0.0)),
                        ssim_weight=float(getattr(self.config.training, "ssim_weight", 0.0)),
                    )
                    loss = loss_dict["total_loss"]

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.ema_helper.update(self.model)

                if self.step % log_freq == 0:
                    total_loss_value = float(loss.item())
                    noise_loss_value = float(loss_dict["noise_loss"].item())
                    psnr_loss_value = float(loss_dict["psnr_loss"].item())
                    ssim_loss_value = float(loss_dict["ssim_loss"].item())
                    psnr_value = float(loss_dict["psnr"].item())
                    ssim_value = float(loss_dict["ssim"].item())

                    print(
                        f"step: {self.step}, "
                        f"total_loss: {total_loss_value:.6f}, "
                        f"noise_loss: {noise_loss_value:.6f}, "
                        f"psnr_loss: {psnr_loss_value:.6f}, "
                        f"ssim_loss: {ssim_loss_value:.6f}, "
                        f"psnr: {psnr_value:.4f}, "
                        f"ssim: {ssim_value:.4f}, "
                        f"data_time: {data_time / (batch_idx + 1):.6f}"
                    )

                    self.metric_tracker.log_train(
                        epoch=epoch,
                        step=self.step,
                        total_loss=total_loss_value,
                        noise_loss=noise_loss_value,
                        psnr_loss=psnr_loss_value,
                        ssim_loss=ssim_loss_value,
                        psnr=psnr_value,
                        ssim=ssim_value,
                    )

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    val_metrics = self.sample_validation_patches(val_loader, self.step)

                    if val_metrics is not None:
                        self.metric_tracker.log_validation(
                            epoch=epoch,
                            step=self.step,
                            psnr=float(val_metrics["psnr"]),
                            ssim=float(val_metrics["ssim"]),
                        )

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    self._save_snapshots(epoch, float(loss.item()))

                data_start = time.time()

    def sample_image(
        self,
        x_cond: torch.Tensor,
        x: torch.Tensor,
        *,
        last: bool = True,
        patch_locs: Optional[Sequence[Tuple[int, int]]] = None,
        patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        sample_steps = max(1, min(int(self.args.sampling_timesteps), self.config.diffusion.num_diffusion_timesteps))
        skip = max(1, self.config.diffusion.num_diffusion_timesteps // sample_steps)
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(
                x,
                x_cond,
                seq,
                self.model,
                self.betas,
                eta=0.0,
                corners=patch_locs,
                p_size=patch_size,
            )
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.0)

        return xs[0][-1] if last else xs

    def sample_validation_patches(self, val_loader: Iterable, step: int) -> Optional[Dict[str, float]]:
        image_folder = Path(self.args.image_folder) / f"{self.config.data.dataset}{self.config.data.image_size}" / str(step)
        image_folder.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            try:
                x, _ = next(iter(val_loader))
            except StopIteration:
                print("Validation loader is empty; skipping validation sampling.")
                return None

            print(f"Processing a single batch of validation images at step: {step}")
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)

            x_cond = x[:, :3, :, :].to(self.device, non_blocking=True)
            x_gt = x[:, 3:, :, :].to(self.device, non_blocking=True)
            x_cond_model = data_transform(x_cond)

            noise = torch.randn(
                n,
                3,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )

            x_pred = self.sample_image(x_cond_model, noise)
            x_pred = inverse_data_transform(x_pred)

            x_pred = x_pred.float()
            x_gt = x_gt.to(device=x_pred.device, dtype=x_pred.dtype, non_blocking=True)

            val_psnr = calculate_psnr_torch(x_pred, x_gt)
            val_ssim = calculate_ssim_torch(x_pred, x_gt)

            val_psnr_value = float(val_psnr.item())
            val_ssim_value = float(val_ssim.item())

            print(f"validation step: {step}, psnr: {val_psnr_value:.4f}, ssim: {val_ssim_value:.4f}")

            x_cond_vis = inverse_data_transform(x_cond_model)
            for idx in range(n):
                utils.logging.save_image(x_cond_vis[idx], str(image_folder / f"{idx}_cond.png"))
                utils.logging.save_image(x_pred[idx], str(image_folder / f"{idx}.png"))

            return {
                "psnr": val_psnr_value,
                "ssim": val_ssim_value,
            }