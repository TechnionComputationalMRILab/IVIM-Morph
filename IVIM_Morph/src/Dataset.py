import torch
from torch.utils.data import Dataset, DataLoader
# import SynthFields
import os
import numpy as np
import torchio
import matplotlib.pyplot as plt

class FetalDataset(Dataset):
    def __init__(self, FetalData, FetalSeg, FetalGA, DATA_ID, Original_phi = None, trans = None, aug_mun = 0):
        self.FetalData = FetalData
        self.FetalSeg = FetalSeg
        self.FetalGA = FetalGA
        self.DATA_ID = DATA_ID
        self.Original_phi = Original_phi
        self.trans = trans
        self.aug_mun = aug_mun

    def __len__(self):
        return self.FetalData.size(0)

    def __getitem__(self, idx):
        FetalData =  self.FetalData[idx]
        FetalSeg = self.FetalSeg[idx]
        FetalGA = self.FetalGA[idx]
        DATA_ID = self.DATA_ID[idx]


        max = torch.quantile(FetalData[ 0, :, :], 0.99)
        FetalData = FetalData / max
        FetalData = torch.where(FetalData > 1, 1, FetalData)

        if self.aug_mun > 0:
            aug_data = []
            FetalData = FetalData.cpu()
            for i in range(self.aug_mun):
                aug = self.trans(FetalData.unsqueeze(-1))
                aug_data.append(aug)
        
            FetalData = torch.stack(aug_data).squeeze(-1)

        if self.Original_phi is not None:
            original_phi = self.Original_phi[idx]
            return FetalData, FetalSeg, FetalGA, DATA_ID, original_phi
        else:
            return FetalData, FetalSeg, FetalGA, DATA_ID
