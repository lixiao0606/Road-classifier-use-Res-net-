'''
    单图测试
'''
import time

import pandas as pd
import torch
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms as transforms
import os


dataset_label = [
    "Asphalt", "Cobblestone", "Gravel",
    "Icy", "Rainy", "Snowy"
]

transform_BZ= transforms.Normalize(
    mean=[0.52200776, 0.5027944, 0.49115777],  # 取决于数据集
    std=[0.09405597, 0.09144473, 0.091840416]
)
#([0.52200776, 0.5027944, 0.49115777], [0.09405597, 0.09144473, 0.091840416])

def padding_black(img,img_size = 256):  # 如果尺寸太小可以扩充
    w, h = img.size
    scale = img_size / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = img_size
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img

if __name__=='__main__':


    img_path = []
    with open('./all.txt', encoding='utf-8') as f:
        # 读取文件
        img_path = f.readlines()

    ground_truth = [ -1 for i in range(len(img_path)) ]
    for index,value in enumerate(img_path) :
        ground_truth[index] = int(img_path[index].split("\t")[1].split("\n")[0])
        img_path[index] = img_path[index].split("\t")[0]

    val_tf = transforms.Compose([  ##简单把图片压缩了变成Tensor模式
        transforms.Resize(256),
        transforms.ToTensor(),
        transform_BZ  # 标准化操作
    ])

    # 如果显卡可用，则用显卡
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    finetune_net1 = resnet18(num_classes=6).to(device)
    state_dict1 = torch.load(r"output/resnet18_z01_best.pth")

    # print("state_dict = ",state_dict)
    finetune_net1.load_state_dict(state_dict1)
    finetune_net1.eval()


    group_id: int
    pred_list = [[] for i in range(len(img_path))]
    time_start = time.time()
    with torch.no_grad():
        for image_i,v in enumerate(img_path):
        # for image_i in range(5):
            img = Image.open(img_path[image_i])  # 打开图片
            img = img.convert('RGB')  # 转换为RGB 格式
            img = padding_black(img, 256)
            img = val_tf(img)
            img_tensor = torch.unsqueeze(img, 0)  # N,C,H,W, ; C,H,W 进行扩张，多一个n的通道
            img_tensor = img_tensor.to(device)
            result = finetune_net1(img_tensor)
            # print("result = ",result.argmax(1))
            #把 结果转换成0-1之间
            pred_softmax = torch.softmax(result, 1).cpu().numpy()
            #把第一轮的结果放在第0，3列
            # pred_list[image_i][0],pred_list[image_i][3] = pred_softmax.tolist()[0][0],pred_softmax.tolist()[0][1]
            # pred_list.append(pred_softmax.tolist()[0])
            group_id = result.argmax(1).item()
            print("预测结果为：", dataset_label[group_id])


            this_img_truth = dataset_label[ground_truth[image_i] ]
            this_img_detect = dataset_label[group_id]
            print("真实标签：\t",this_img_truth , "预测结果为：\t",this_img_detect)

    time_end = time.time()
    print(f"总耗时: {(time_end - time_start)}s")
    print(f"平均每张图片耗时: {(time_end - time_start)*1000/len(img_path)}ms")
    print(f"帧率: {len(img_path)/(time_end - time_start) }")
    file_name_list = dataset_label
    df_pred = pd.DataFrame(data=pred_list, columns=file_name_list)
    # print(df_pred)
    df_pred.to_csv('pred_result.csv', encoding='gbk', index=False)
