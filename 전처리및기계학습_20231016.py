import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

print(datetime.now())

# 데이터 불러오기
data = pd.read_csv("2023-10-10-롯데카드_소비_데이터.csv", encoding='cp949')  # 데이터 파일명을 지정

# 결측치 처리
imputer = SimpleImputer(strategy="median")
data["sl_am"] = imputer.fit_transform(data[["sl_am"]])

# 이상치 처리 (예: Z-score 기반)
z_scores = np.abs((data["sl_am"] - data["sl_am"].mean()) / data["sl_am"].std())
data = data[(z_scores < 3)]

# 정규화 (예: Z-score 정규화)
scaler = StandardScaler()
data[["sl_am", "sl_ct"]] = scaler.fit_transform(data[["sl_am", "sl_ct"]])

# 데이터 분석 및 기술 개발 부분은 문제에 따라 추가적인 코드를 작성해야 합니다.

# 전처리된 데이터를 파일로 저장 (옵션)
data.to_csv("preprocessed_data.csv", index=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6 * 13 * 13)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 데이터셋을 두 번 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
print('Finished Training')
print(datetime.now())