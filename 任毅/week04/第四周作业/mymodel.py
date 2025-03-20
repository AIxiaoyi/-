import torch as torch
import torch.nn as nn


class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(4096,2048)
        self.bn = nn.BatchNorm1d(2048)
        self.ln2 = nn.Linear(2048,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.ln3 = nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.ln4 = nn.Linear(512,256)
        self.bn3 = nn.BatchNorm1d(256)
        self.ln5 = nn.Linear(256,128)
        self.bn4 = nn.BatchNorm1d(128)
        self.ln6 = nn.Linear(128,64)
        self.bn5 = nn.BatchNorm1d(64)
        self.ln7 = nn.Linear(64,40)
        self.dp = nn.Dropout(p=0.5)
        self.rl = nn.ReLU()
        
        
        
    
    def forward(self,x):
        out = self.ln1(x)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn(out)
        out =self.dp(out)
        out =self.ln2(out)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn1(out)
        out =self.ln3(out)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn2(out)
        out =self.ln4(out)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn3(out)
        out =self.ln5(out)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn4(out)
        out =self.ln6(out)
        out =self.rl(out)
        out = self.dp(out)
        out = self.bn5(out)
        out =self.ln7(out)
    
        return out
