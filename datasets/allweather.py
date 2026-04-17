import os
import random
import numpy as np
import torch
import torchvision
import torch.utils.data
from PIL import Image


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


class AllWeather:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def get_loaders(self, parse_patches=True, validation='snow'):
        if validation not in ['fog', 'rainfog', 'snow']:
            raise ValueError(
                f"Unsupported validation split: {validation}. "
                f"Expected one of ['fog', 'rainfog', 'snow']"
            )

        print("=> training on unified allweather dataset...")
        print(f"=> evaluating {validation} test set...")

        train_dataset = AllWeatherDataset(
            root=self.config.data.data_dir,
            patch_size=self.config.data.image_size,
            n=self.config.training.patch_n,
            transforms=self.transforms,
            split='train',
            weather=None,
            parse_patches=parse_patches
        )

        val_dataset = AllWeatherDataset(
            root=self.config.data.data_dir,
            patch_size=self.config.data.image_size,
            n=self.config.training.patch_n,
            transforms=self.transforms,
            split='test',
            weather=validation,
            parse_patches=parse_patches
        )

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, root, patch_size, n, transforms, split='train', weather=None, parse_patches=True):
        super().__init__()
        self.root = root
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.split = split
        self.weather = weather
        self.parse_patches = parse_patches
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f'No image pairs found for split="{self.split}", weather="{self.weather}"')

    @staticmethod
    def _is_image_file(filename):
        return filename.lower().endswith(IMG_EXTENSIONS)

    @staticmethod
    def _list_image_files(folder):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Folder not found: {folder}')
        files = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f)) and AllWeatherDataset._is_image_file(f)
        ]
        return files

    @staticmethod
    def _stem(path):
        return os.path.splitext(os.path.basename(path))[0]

    def _pair_from_dirs(self, input_dir, gt_dir):
        input_files = self._list_image_files(input_dir)
        gt_files = self._list_image_files(gt_dir)

        if len(input_files) != len(gt_files):
            raise RuntimeError(
                f'Mismatched pair counts:\n'
                f'input_dir={input_dir} -> {len(input_files)} files\n'
                f'gt_dir={gt_dir} -> {len(gt_files)} files'
            )

        input_map = {self._stem(p): p for p in input_files}
        gt_map = {self._stem(p): p for p in gt_files}

        if set(input_map.keys()) == set(gt_map.keys()):
            common_keys = sorted(input_map.keys())
            return [(input_map[k], gt_map[k]) for k in common_keys]

        # Fallback: sorted positional pairing
        return list(zip(input_files, gt_files))

    def _build_samples(self):
        samples = []

        if self.split == 'train':
            train_root = os.path.join(self.root, 'data', 'allweather')
            weather_types = ['fog', 'rainfog', 'snow']

            for weather in weather_types:
                input_dir = os.path.join(train_root, weather, 'input')
                gt_dir = os.path.join(train_root, weather, 'gt')
                samples.extend(self._pair_from_dirs(input_dir, gt_dir))

        elif self.split == 'test':
            if self.weather not in ['fog', 'rainfog', 'snow']:
                raise ValueError(
                    f'Unsupported test weather: {self.weather}. '
                    f'Expected one of ["fog", "rainfog", "snow"]'
                )

            test_root = os.path.join(self.root, 'data', self.weather, 'test')
            input_dir = os.path.join(test_root, 'input')
            gt_dir = os.path.join(test_root, 'gt')
            samples = self._pair_from_dirs(input_dir, gt_dir)

        else:
            raise ValueError(f'Unsupported split: {self.split}')

        return samples

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size

        if w == tw and h == th:
            return [0] * n, [0] * n, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    @staticmethod
    def _resize_filter():
        if hasattr(Image, "Resampling"):
            return Image.Resampling.LANCZOS
        return Image.ANTIALIAS

    def get_images(self, index):
        input_path, gt_path = self.samples[index]
        img_id = self._stem(input_path)

        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_patches = self.n_random_crops(input_img, i, j, h, w)
            gt_patches = self.n_random_crops(gt_img, i, j, h, w)

            outputs = [
                torch.cat([self.transforms(input_patches[k]), self.transforms(gt_patches[k])], dim=0)
                for k in range(self.n)
            ]
            return torch.stack(outputs, dim=0), img_id

        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024

        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))

        resize_mode = self._resize_filter()
        input_img = input_img.resize((wd_new, ht_new), resize_mode)
        gt_img = gt_img.resize((wd_new, ht_new), resize_mode)

        return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.samples)