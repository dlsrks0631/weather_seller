#정석님 코드를 수정했습니다. 아나콘다로 파이토치 실행했습니다.
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

#수정되면 다음과 같이 만들 수 있습니다.

print(datetime.now())

chunk_size = 500000
file_path = r"/Users/kimjehyeon/Desktop/2023_3_2/캡스톤 - Weather Seller/롯데카드데이터.csv"
chunks = pd.read_csv(file_path, encoding='utf-8', chunksize=chunk_size)
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

# 처음 작성을 위한 플래그 설정
first_write = True
output_file = "preprocessed_data.csv"

for chunk in chunks:
    chunk["sl_am"] = imputer.fit_transform(chunk[["sl_am"]])
    chunk["sl_ct"] = imputer.fit_transform(chunk[["sl_ct"]])
    z_scores = np.abs((chunk["sl_am"] - chunk["sl_am"].mean()) / chunk["sl_am"].std())
    chunk = chunk[(z_scores < 3)]
    chunk[["sl_am", "sl_ct"]] = scaler.fit_transform(chunk[["sl_am", "sl_ct"]])
    
    # 각 청크를 처리한 후 바로 파일에 저장
    if first_write:
        chunk.to_csv(output_file, index=False)
        first_write = False
    else:
        chunk.to_csv(output_file, mode='a', header=False, index=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

for epoch in range(2):
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
