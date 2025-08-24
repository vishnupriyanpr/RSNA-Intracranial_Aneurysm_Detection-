import numpy as np
import SimpleITK as sitk

def sitk_resample(img: sitk.Image, new_spacing_xyz):
    original_spacing = np.array(list(img.GetSpacing()))  # (x,y,z)
    original_size = np.array(list(img.GetSize()), dtype=float)  # (x,y,z)
    new_spacing = np.array(new_spacing_xyz, dtype=float)
    new_size = np.maximum(1, np.round(original_size * (original_spacing / new_spacing))).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return resampler.Execute(img)

def cta_window_minmax(arr: np.ndarray, wl_low=-200, wl_high=700):
    arr = np.clip(arr, wl_low, wl_high)
    arr = (arr - wl_low) / (wl_high - wl_low + 1e-6)
    return arr.astype(np.float32)

def zscore(arr: np.ndarray):
    m = arr.mean()
    s = arr.std() + 1e-6
    return ((arr - m) / s).astype(np.float32)
