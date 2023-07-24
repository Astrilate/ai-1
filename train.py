import torch
import torchvision
from torch.utils.data import DataLoader
import datetime
from matplotlib import pyplot as plt


# 加载处理手写数字数据集
train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        # download=True
)

test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        # download=True
)

# 为训练数据和测试数据加数据加载器
train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=100,
        shuffle=True  # 每次洗牌
)

test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=10000,
        shuffle=True
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 训练模型设置
class handwriting(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size):  # 数据维度， 中间层神经元数量， 输出维度
        super(handwriting, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(0.2)  # 添加一个Dropout层
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)  # 增加一个隐藏层
        self.linear3 = torch.nn.Linear(hidden_size, out_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        # out = self.dropout(out)  # 在第一个隐藏层后应用Dropout
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    model = handwriting(28*28, 500, 10).to(device)
    lossF = torch.nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    # print(list(model.parameters())[0].shape)

    # 训练模型
    st = datetime.datetime.now()
    for epoch in range(5):  # 把整个数据集训练5次
        for i, (features, labels) in enumerate(train_dataloader):
            # features = torch.where(features > 0.82, torch.ones_like(features), torch.zeros_like(features))  # 二值化？
            # print(features[0][0], features[0][0].shape)
            # plt.imshow(features[0][0], cmap="gray")
            # plt.axis("off")
            # plt.show()
            features = features.view(-1, 28*28).to(device)  # 展平
            labels = labels.to(device)
            predict = model(features)  # 预测值
            ls = lossF(predict, labels)  # 计算损失
            ls.backward()  # 向后传播  /  求到模型参数的梯度
            optimizer.step()  # 参数更新
            optimizer.zero_grad()  # 去掉所有参数累积的梯度值
            if i % 100 == 0:
                print(f"loss = {ls:.8f}")
    et = datetime.datetime.now()
    Time = (et - st).seconds
    print(f"训练耗时{Time}秒")  # 共5轮，每轮60000条数据，每批提供100个

    # 存储模型
    # torch.save(model.state_dict(), "./hwn1.pth")
