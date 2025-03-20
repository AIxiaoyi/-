import torch
from torch.utils.data import Dataset, DataLoader




class CustomDataset(Dataset):
    def __init__(self,features,labels):
        super().__init__()
        # TODO: 1. 完成数据集的初始化
        self.features =torch.tensor(features)
        self.labels =torch.tensor(labels)

    def __len__(self):
        # TODO: 2. 实现数据集的长度
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

        return x,y
