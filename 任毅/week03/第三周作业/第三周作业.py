import torch as plt
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import transforms
from collections import OrderedDict


# 定义超参数
LR = 1e-3
epochs =1000
BATCH_SIZE = 100

data_train = KMNIST('data_kmnist', train=True, download=True, transform=transforms.ToTensor())
test_train = KMNIST('data_kmnist', train=False, download=True, transform=transforms.ToTensor())

train_dl =DataLoader(data_train,batch_size=BATCH_SIZE, shuffle=True)



model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(28 * 28, 256)),  # 输入层 -> 隐藏层1
    ('sigmoid1', nn.Sigmoid()),

    ('fc2',nn.Linear(256, 128)),# 隐藏层1 -> 隐藏层2
    ('sigmoid2', nn.Sigmoid()),

    ('fc3',nn.Linear(128, 64)), #隐藏层2 -> 隐藏层3
    ('sigmoid3', nn.Sigmoid()),

    ('fc4',nn.Linear(64, 10)),  #隐藏层3 -> 输出层
    ('output', nn.Softmax(dim=1))   # 将输出层归一化
])
)
device = plt.device("cuda" if plt.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()  #交叉熵损失函数
# 优化器（模型参数更新）
optimizer = plt.optim.SGD(model.parameters(), lr=LR)


for epoch in range(epochs):
    # 遍历数据
    for data, target in train_dl:
        data,target = data.to(device),target.to(device)
        # 前向计算
        output = model(data.reshape(-1, 28*28))
        # 计算损失
        loss = loss_fn(output, target)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

    # 每5个epoch打印一次损失
    if epoch % 50 == 0:
        with open("losses.txt", "a") as file:
            file.write(f'Epoch: {epoch} Loss: {loss.item()}\n')
        print(f'Epoch: {epoch} Loss: {loss.item()}')









