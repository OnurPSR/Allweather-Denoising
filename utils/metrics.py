import cv2
import numpy as np
import torch
import torch.nn.functional as F

# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR


def calculate_psnr(img1, img2, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img



def calculate_psnr_torch(img1, img2, max_val=1.0, eps=1e-8, reduction='mean'):
    """Calculate PSNR for torch tensors in BCHW format.

    Args:
        img1 (Tensor): Predicted image tensor with range [0, 1].
        img2 (Tensor): Ground-truth image tensor with range [0, 1].
        max_val (float): Maximum possible value in the image range.
        eps (float): Numerical stability term.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        Tensor: Reduced PSNR value or per-image PSNR values.
    """
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if img1.ndim != 4:
        raise ValueError(f'Expected BCHW tensors, but got shape {img1.shape}.')

    img1 = img1.float()
    img2 = img2.float()
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + eps))

    if reduction == 'none':
        return psnr
    if reduction == 'sum':
        return psnr.sum()
    if reduction == 'mean':
        return psnr.mean()
    raise ValueError(f'Unsupported reduction: {reduction}')


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def calculate_ssim_torch(img1, img2, window_size=11, sigma=1.5, data_range=1.0, eps=1e-8, reduction='mean'):
    """Calculate SSIM for torch tensors in BCHW format.

    Args:
        img1 (Tensor): Predicted image tensor with range [0, 1].
        img2 (Tensor): Ground-truth image tensor with range [0, 1].
        window_size (int): Gaussian kernel size.
        sigma (float): Gaussian sigma.
        data_range (float): Dynamic range of the image.
        eps (float): Numerical stability term.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        Tensor: Reduced SSIM value or per-image SSIM values.
    """
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if img1.ndim != 4:
        raise ValueError(f'Expected BCHW tensors, but got shape {img1.shape}.')

    img1 = img1.float()
    img2 = img2.float()
    _, channels, height, width = img1.shape

    real_window = min(window_size, height, width)
    if real_window % 2 == 0:
        real_window -= 1
    if real_window < 1:
        raise ValueError(f'Invalid window size computed from shape {img1.shape}.')

    window = _gaussian_window(real_window, sigma, channels, img1.device, img1.dtype)
    pad = real_window // 2

    img1_pad = F.pad(img1, (pad, pad, pad, pad), mode='reflect')
    img2_pad = F.pad(img2, (pad, pad, pad, pad), mode='reflect')

    mu1 = F.conv2d(img1_pad, window, groups=channels)
    mu2 = F.conv2d(img2_pad, window, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (pad, pad, pad, pad), mode='reflect'), window, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (pad, pad, pad, pad), mode='reflect'), window, groups=channels) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (pad, pad, pad, pad), mode='reflect'), window, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps
    )
    ssim = ssim_map.mean(dim=(1, 2, 3))

    if reduction == 'none':
        return ssim
    if reduction == 'sum':
        return ssim.sum()
    if reduction == 'mean':
        return ssim.mean()
    raise ValueError(f'Unsupported reduction: {reduction}')