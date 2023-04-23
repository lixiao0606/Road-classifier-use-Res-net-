from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
# 进度条
from tqdm import tqdm

# 压缩图片尺寸为短边224，为了运算速度
transform = T.Compose([
     T.RandomResizedCrop(256),
     T.ToTensor(),
])

def getStat(train_data):
    # 定义一个数据加载器，传入训练数据，bs1，不洗牌，不多线程
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    mean = torch.zeros(3) #均值
    std = torch.zeros(3) #方差
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean() # N, C, H ,W：C为3个通道
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'enhance_dataset', transform=transform)
    print(getStat(train_dataset))
    # ([0.52165467, 0.5025819, 0.4910438], [0.09407499, 0.09147017, 0.0918781])