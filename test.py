import torch
import matplotlib.pyplot as plt
from train import device, handwriting, test_dataloader

import numpy as np
from PIL import Image

# 加载模型
model = handwriting(28*28, 500, 10).to(device)
model.load_state_dict(torch.load("./hwn.pth"))

# 使用Mnist数据集中测试数据
# 迭代test_dataloader并显示1个样本
for i, (features, labels) in enumerate(test_dataloader):
    if i > 0:  # 仅显示1个批次，1个批次batch包含100个数据，但是通常只使用这100张的前几张比如下面的是前6张
        break

    # 展示图片
    axes = plt.subplots(1, 9)[1]  # 创建一个包含1行2列的子图对象，plt.subplots(1, 2)[0]为整个图形的对象，axes为包含的子图的数组，
    # 可通过此数组对每张子图的内容进行设置
    for j, ax in enumerate(axes):
        ax.imshow(features[j][0], cmap="gray")  # j为每个图像在该批次(100张)，均为(0~5)的序号

    # 测试图片
    # features = torch.where(features > 0.82, torch.ones_like(features), torch.zeros_like(features))
    features = features.view(-1, 28*28)  # 展平测试图片，才能放进模型进行测试
    print(features.shape)
    with torch.no_grad():
        predict = model(features)  # 模型测试
        Indices = torch.argmax(predict, dim=1)  # 获取每行（浮点数）最大值的索引
        print(f"正确率： {(torch.eq(Indices, labels).sum().item() / len(labels)):.2%}")  # 计算bool张量中True个数
        print("预测：", Indices.tolist())
        print("答案：", labels.tolist())
    plt.show()  # 展示前2张图


# 测试自己的图片
correct_total = 0
total = 0
for k in range(10):
    image = Image.open(f"./8pixel/{k}.jpg")
    image = image.resize((28, 28))
    gray_image = image.convert("L")  # 转换为灰度图像
    inverted_image = Image.eval(gray_image, lambda x: 255 if x < 210 else 0)  # 二值化
    np_image = np.array(inverted_image) / 255.0  # 图片转为 NumPy 数组并标准化
    features = torch.from_numpy(np_image).float()
    # print(features.shape)  # [28, 28]才可以展示
    # plt.imshow(features, cmap="gray")
    # plt.axis("off")
    # plt.show()
    features = features.view(-1, 28*28)
    # print(features, features.shape)  # [1, 784]就可以训练了
    with torch.no_grad():
        predict = model(features)
        Indices = torch.argmax(predict, dim=1)
        print("答案为：", Indices.tolist()[0])
        if Indices.tolist()[0] == k:
            correct_total += 1
        total += 1
print(f"{correct_total / total:.2%}")
print(correct_total, total)
