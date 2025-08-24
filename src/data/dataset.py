import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from .preprocessing import sitk_resample, cta_window_minmax, zscore
from .transforms import train_augments

class RSNAAneurysmDataset(Dataset):
    def __init__(self, df, img_root, targets, spacing_cfg, window_cfg, mode="train"):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.targets = targets
        self.spacing_cfg = spacing_cfg
        self.window_cfg = window_cfg
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _dicom_series_path(self, uid):
        # Expected: series/<SeriesInstanceUID>/
        return os.path.join(self.img_root, "series", str(uid))

    def _load_sitk_series(self, series_dir):
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(series_dir)
        reader.SetFileNames(files)
        return reader.Execute()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["SeriesInstanceUID"]
        modality = str(row["Modality"])
        series_dir = self._dicom_series_path(uid)
        img = self._load_sitk_series(series_dir)

        # spacing by modality
        if "CTA" in modality:
            spacing = self.spacing_cfg["CTA"]
        elif "MRA" in modality:
            spacing = self.spacing_cfg["MRA"]
        else:
            spacing = self.spacing_cfg["MRI"]

        img = sitk_resample(img, spacing)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X) in SITK numpy

        if "CTA" in modality:
            wl_low, wl_high = self.window_cfg["CTA"]
            vol = cta_window_minmax(vol, wl_low, wl_high)
        else:
            vol = zscore(vol)

        if self.mode == "train":
            vol = train_augments(vol)

        vol = np.expand_dims(vol, 0)  # (C=1,Z,Y,X)
        x = torch.from_numpy(vol)
        y = torch.tensor([float(row[t]) for t in self.targets], dtype=torch.float32)
        return x, y, uid
