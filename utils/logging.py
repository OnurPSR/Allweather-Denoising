import torch
import shutil
import os
import torchvision.utils as tvu
import tempfile

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


def load_checkpoint(path, device="cpu", *, mmap=False):
    return torch.load(
        path,
        map_location=device,
        weights_only=True,
        mmap=mmap,
    )