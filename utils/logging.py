from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not filename.endswith(".pth.tar"):
        filename = filename + ".pth.tar"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(filename),
        prefix=".tmp_ckpt_",
        suffix=".pth.tar",
    )
    os.close(fd)

    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class CheckpointSaver:
    def __init__(self, save_dir: str | os.PathLike[str], prefix: str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    @property
    def last_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_last.pth.tar"

    @property
    def best_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_best.pth.tar"

    def save_last(self, state: Dict[str, Any]) -> str:
        save_checkpoint(state, str(self.last_path))
        return str(self.last_path)

    def save_best(self, state: Dict[str, Any], current_loss: float, best_loss: float) -> Tuple[bool, float, str]:
        improved = current_loss < best_loss
        updated_best_loss = current_loss if improved else best_loss

        if improved:
            save_checkpoint(state, str(self.best_path))

        return improved, updated_best_loss, str(self.best_path)


class LiveMetricTracker:
    """
    Appends training/validation/snapshot metrics to a CSV file and refreshes PNG graphs.
    This is intentionally simple and file-based, so you can monitor it live without tensorboard.
    """

    FIELDNAMES = [
        "timestamp",
        "epoch",
        "step",
        "phase",
        "total_loss",
        "noise_loss",
        "psnr_loss",
        "ssim_loss",
        "psnr",
        "ssim",
        "current_loss",
        "best_loss",
        "is_best",
    ]

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        prefix: str,
        plot_every_n_events: int = 1,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.prefix = prefix
        self.csv_path = self.log_dir / f"{self.prefix}_metrics.csv"
        self.loss_plot_path = self.log_dir / f"{self.prefix}_losses.png"
        self.psnr_plot_path = self.log_dir / f"{self.prefix}_psnr.png"
        self.ssim_plot_path = self.log_dir / f"{self.prefix}_ssim.png"

        self.plot_every_n_events = max(1, int(plot_every_n_events))
        self.event_count = 0
        self.rows: List[Dict[str, Any]] = []

        self._bootstrap_csv()

    def _bootstrap_csv(self) -> None:
        if self.csv_path.exists():
            self._load_existing_csv()
        else:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def _to_float_or_nan(self, value: Any) -> float:
        if value is None:
            return math.nan
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return math.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    def _to_int_or_zero(self, value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _to_bool_int(self, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": self._to_float_or_nan(row.get("timestamp", time.time())),
            "epoch": self._to_int_or_zero(row.get("epoch", 0)),
            "step": self._to_int_or_zero(row.get("step", 0)),
            "phase": str(row.get("phase", "")),
            "total_loss": self._to_float_or_nan(row.get("total_loss")),
            "noise_loss": self._to_float_or_nan(row.get("noise_loss")),
            "psnr_loss": self._to_float_or_nan(row.get("psnr_loss")),
            "ssim_loss": self._to_float_or_nan(row.get("ssim_loss")),
            "psnr": self._to_float_or_nan(row.get("psnr")),
            "ssim": self._to_float_or_nan(row.get("ssim")),
            "current_loss": self._to_float_or_nan(row.get("current_loss")),
            "best_loss": self._to_float_or_nan(row.get("best_loss")),
            "is_best": self._to_bool_int(row.get("is_best", 0)),
        }

    def _load_existing_csv(self) -> None:
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.rows = [self._normalize_row(row) for row in reader]

    def _append_row(self, row: Dict[str, Any]) -> None:
        normalized = self._normalize_row(row)
        self.rows.append(normalized)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(normalized)

        self.event_count += 1
        if self.event_count % self.plot_every_n_events == 0:
            self.refresh_plots()

    def log_train(
        self,
        *,
        epoch: int,
        step: int,
        total_loss: float,
        noise_loss: float,
        psnr_loss: float,
        ssim_loss: float,
        psnr: float,
        ssim: float,
    ) -> None:
        self._append_row(
            {
                "timestamp": time.time(),
                "epoch": epoch,
                "step": step,
                "phase": "train",
                "total_loss": total_loss,
                "noise_loss": noise_loss,
                "psnr_loss": psnr_loss,
                "ssim_loss": ssim_loss,
                "psnr": psnr,
                "ssim": ssim,
                "current_loss": "",
                "best_loss": "",
                "is_best": 0,
            }
        )

    def log_validation(
        self,
        *,
        epoch: int,
        step: int,
        psnr: float,
        ssim: float,
    ) -> None:
        self._append_row(
            {
                "timestamp": time.time(),
                "epoch": epoch,
                "step": step,
                "phase": "val",
                "total_loss": "",
                "noise_loss": "",
                "psnr_loss": "",
                "ssim_loss": "",
                "psnr": psnr,
                "ssim": ssim,
                "current_loss": "",
                "best_loss": "",
                "is_best": 0,
            }
        )

    def log_snapshot(
        self,
        *,
        epoch: int,
        step: int,
        current_loss: float,
        best_loss: float,
        is_best: bool,
    ) -> None:
        self._append_row(
            {
                "timestamp": time.time(),
                "epoch": epoch,
                "step": step,
                "phase": "snapshot",
                "total_loss": current_loss,
                "noise_loss": "",
                "psnr_loss": "",
                "ssim_loss": "",
                "psnr": "",
                "ssim": "",
                "current_loss": current_loss,
                "best_loss": best_loss,
                "is_best": int(is_best),
            }
        )

    def _extract_series(self, phase: str, metric_key: str) -> Tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []

        for row in self.rows:
            if row["phase"] != phase:
                continue
            value = row.get(metric_key, math.nan)
            if isinstance(value, float) and math.isfinite(value):
                xs.append(int(row["step"]))
                ys.append(float(value))

        return xs, ys

    def _save_figure_atomic(self, fig, path: Path) -> None:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".tmp_plot_",
            suffix=path.suffix,
        )
        os.close(fd)

        try:
            fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _plot_losses(self) -> None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for phase, key, label in [
            ("train", "total_loss", "train total_loss"),
            ("train", "noise_loss", "train noise_loss"),
            ("train", "psnr_loss", "train psnr_loss"),
            ("train", "ssim_loss", "train ssim_loss"),
            ("snapshot", "best_loss", "best_loss"),
        ]:
            xs, ys = self._extract_series(phase, key)
            if xs:
                ax.plot(xs, ys, label=label)

        ax.set_title("Loss Curves")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        self._save_figure_atomic(fig, self.loss_plot_path)
        plt.close(fig)

    def _plot_psnr(self) -> None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for phase, label in [
            ("train", "train PSNR"),
            ("val", "val PSNR"),
        ]:
            xs, ys = self._extract_series(phase, "psnr")
            if xs:
                ax.plot(xs, ys, label=label)

        ax.set_title("PSNR Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("PSNR")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        self._save_figure_atomic(fig, self.psnr_plot_path)
        plt.close(fig)

    def _plot_ssim(self) -> None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for phase, label in [
            ("train", "train SSIM"),
            ("val", "val SSIM"),
        ]:
            xs, ys = self._extract_series(phase, "ssim")
            if xs:
                ax.plot(xs, ys, label=label)

        ax.set_title("SSIM Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("SSIM")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        self._save_figure_atomic(fig, self.ssim_plot_path)
        plt.close(fig)

    def refresh_plots(self) -> None:
        self._plot_losses()
        self._plot_psnr()
        self._plot_ssim()


def load_checkpoint(path, device="cpu", *, mmap=False):
    return torch.load(
        path,
        map_location=device,
        weights_only=True,
        mmap=mmap,
    )