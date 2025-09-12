"""
Ultrasound Image Denoising Utilities
"""


from typing import Optional
from enum import Enum

import numpy as np
import cv2
from numpy.typing import ArrayLike


class DenoisingModes(Enum):
    """
    Denoising Modes Enum
    """
    NOISE = 0
    SIGNAL = 1


def calculate_2dft(img: ArrayLike):
    """
    Calculate the 2-Dimentional Discrete Fourier Transform
    """
    fft_img = np.fft.ifftshift(img)
    fft_img = np.fft.fft2(fft_img)
    fft_img = np.fft.fftshift(fft_img)
    return fft_img


def calculate_db_magnitude_spectrum(img: ArrayLike):
    """
    Calculate the Spectrum Magnitude, in db, of a given image
    """
    img_fft = calculate_2dft(img)
    img_ms = 20*np.log10(np.abs(img_fft))
    return img_ms


def calculate_2dift(img_fft: ArrayLike):
    """
    Calculate the inverse of the 2-Dimensional Discrete Fourier Transform
    """
    ift = np.fft.ifftshift(img_fft)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    ift = ift.real
    return ift


def create_noise_mask(img, thresh_list=None):
    """
    Create an image mask for noise
    """
    img_ms = calculate_db_magnitude_spectrum(img)
    if thresh_list is None:
        thresh_min = np.min(img_ms)
        thresh_max = np.max(img_ms)
    else:
        thresh_min = thresh_list[0]
        thresh_max = thresh_list[1]
    mask = img_ms
    mask[mask<thresh_min] = 1.0
    mask[mask>thresh_max] = 1.0
    mask[mask != 1.0] = 0.0
    return mask


def create_signal_mask(
    img: ArrayLike,
    thresh_list: Optional[list[int|float|np.number]]
):
    """
    Create an image mask for signal
    """
    img_ms = calculate_db_magnitude_spectrum(img)
    if thresh_list is None:
        thresh_min = np.min(img_ms)
        thresh_max = np.max(img_ms)
    else:
        thresh_min = thresh_list[0]
        thresh_max = thresh_list[1]
    mask = img_ms
    mask[mask<thresh_min] = 0.0
    mask[mask>thresh_max] = 0.0
    mask[mask != 0.0] = 1.0
    return mask


def denoising_fft(
    img: ArrayLike,
    thresh_list: Optional[list[int|float|np.number]],
    mode: DenoisingModes
):
    """
    Image denoising using FFT
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    match mode:
        case DenoisingModes.NOISE:
            mask = create_noise_mask(img, thresh_list)
            img_fft = calculate_2dft(img)
            noise = calculate_2dift(img_fft*mask)
            dummy0 = img - noise
            dummy1 = dummy0
            dummy1[dummy1 < 0] = 0
            dummy2 = np.uint8(dummy1)
            return dummy2
        case DenoisingModes.SIGNAL:
            mask = create_signal_mask(img, thresh_list)
            img_fft = calculate_2dft(img)
            signal = calculate_2dift(img_fft*mask)
            signal[signal < 0] = 0
            signal = np.uint8(signal)
            return signal
        case _:
            raise ValueError("mode is not valid")


def denoising(
    img: ArrayLike,
    alpha: float = 0.5,
    beta: float = 0.5
):
    """
    Main Denoising Function
    """
    img = np.array(img)
    equ = cv2.equalizeHist(img) # pylint: disable=no-member
    img_ms = calculate_db_magnitude_spectrum(equ)
    umbral_inferior = np.uint16(np.mean(img_ms)+alpha*np.std(img_ms))
    umbral_superior = np.uint16(np.max(img_ms)-beta*np.std(img_ms))
    new_img = denoising_fft(equ,[umbral_inferior,umbral_superior],mode=DenoisingModes.SIGNAL)
    return new_img


# pylint: disable=no-member
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    load_dotenv()
    DATASET_DIR = os.getenv("DATASET_DIR")
    FILE_PATH = os.path.join(*[DATASET_DIR, "Training and Validation Set", "pd_25.png"])
    image = cv2.imread(FILE_PATH, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    equalized_image = cv2.equalizeHist(image)
    new_image = denoising(image, alpha=0.5, beta=0.5)
    fig = plt.figure(figsize=(20, 15), num="Denoising Test")
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Original Image")
    ax.imshow(image, cmap='gray')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Equalized Image")
    ax.imshow(equalized_image, cmap='gray')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Denoised Image")
    ax.imshow(new_image, cmap='gray')
    plt.show()
