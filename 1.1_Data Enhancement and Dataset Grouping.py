# Data Enhancement and Dataset Grouping

import os
import random
import cv2
import numpy as np
from tqdm import tqdm

# 参数
dataset_label = [
    "Asphalt", "Cobblestone", "Gravel", "Icy", "Rainy", "Snowy"
]

# 采用 k 折交叉验证 ，k取5
val_k = 6
group_name = ["Group" + str(val_k) for i in range(val_k)]

# root
project_root = os.getcwd()
dataset_root =os.path.join(project_root,r"dataset")
enhance_dataset_root =os.path.join(project_root,r"enhance_dataset")
if os.path.isdir(enhance_dataset_root) == False:
    os.makedirs(enhance_dataset_root)
for i,v in enumerate(dataset_label):
    temp = os.path.join(enhance_dataset_root,v)
    if os.path.isdir(temp) == False:
        os.makedirs(temp)
dataset_group_root = os.path.join(project_root,r"dataset_group")
if os.path.isdir(dataset_group_root) == False:
    os.makedirs(dataset_group_root)


train_txt_prefix = os.path.join(dataset_group_root,"train_group_")
val_txt_prefix = os.path.join(dataset_group_root,"val_group_")






# 图片亮度改变
def gamma_trans(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


def liner_trans(img,gamma):#gamma大于1时图片变亮，小于1图片变暗
    img=np.float32(img)*gamma//1
    img[img>255]=255
    img=np.uint8(img)
    return img

#增加噪声
def add_noise(img,lever):
    # 获取图像尺寸
    height, width = img.shape[:2]

    # 定义噪声参数
    mean = 0
    var = 50 # 均值0.1*random.random()
    sigma = var ** (0.6 + lever*0.1) # 方差var ** 0.5

    # 生成高斯噪声
    gauss = np.random.normal(mean, sigma, (height, width, 3))

    # 将图像和噪声相加
    noisy_img = np.clip(img[:,:,:3] + gauss,0,255)
    return noisy_img


# 水平翻转
def Horizontal(image):
    return cv2.flip(image,1,dst=None) #水平镜像

# 垂直翻转
def Vertical(image):
    return cv2.flip(image,0,dst=None) #垂直镜像

# 随即裁剪
def RamdonCrop(image):
    h = len(image)
    w = len(image[0])
    h1 = random.choice(range(0, h // 2))
    w1 = random.choice(range(0, w // 2))
    image = image[h1:h//2+h1,w1:w//2+w1]
    image = cv2.resize(image,(w,h))
    return image

# 数据增强与数据集制作:图片路径,类别标签，训练集图片路径保存的列表，验证集……(path.jpg,3,[],[])
def Image_enhance(image,id,train_place,val_place):

    classification = dataset_label[id]
    # print("图像增强："+ image)
    # 这样读取和保存可以防止中文报错
    img_i = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)  # 读取图片
    temp_file_name = image.split("\\")[-1].split(".")[-2]


    # 原图
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_original.jpg")
    cv2.imencode('.jpg', img_i)[1].tofile(temp_path)  # 保存图片
    val_place.append(temp_path+'\t' + str(id) + '\n')
    train_place.append(temp_path+'\t' + str(id) + '\n')

    # 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
    img_corrected = gamma_trans(img_i, 0.45)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_correct_0.45.jpg")
    cv2.imencode('.jpg', img_corrected)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')


    img_corrected = gamma_trans(img_i, 0.7 + random.random() * 0.1)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_correct_ram1.jpg")
    cv2.imencode('.jpg', img_corrected)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')


    img_corrected = gamma_trans(img_i, 1.7 + random.random() * 0.2)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_correct_ram2.jpg")
    cv2.imencode('.jpg', img_corrected)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')


    img_corrected = gamma_trans(img_i, 2.2)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_correct_ram2.2.jpg")
    cv2.imencode('.jpg', img_corrected)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')


    # img_RamdonCrop = RamdonCrop(img_i)
    # cv2.imencode('.jpg', img_RamdonCrop)[1].tofile(os.path.join(save_path, file_i.split('.')[-2] + "_cropped.jpg"))  # 保存图片

    img_noise = add_noise(img_i, 1)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_img_noise.jpg")
    cv2.imencode('.jpg', img_noise)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')


    img_noise = add_noise(img_i, 2)
    temp_path = os.path.join(enhance_dataset_root,classification, temp_file_name + "_img_noise2.jpg")
    cv2.imencode('.jpg', img_noise)[1].tofile(temp_path)
    train_place.append(temp_path+'\t' + str(id) + '\n')



if __name__ == '__main__':

    #生成文件夹enhance_dataset装所有的增强过后的图片
    # 生成txt文件 共10个：
    #   5个txt装训练的图片的位置（增强的）
    #       train_data_group0.txt
    #   5个txt装验证的图片的位置
    #       val_data_group0.txt


    # 原数据集ori_data,双重列表：6个类，以及每个类下所有文件的路径
    #路径用绝对路径
    # 分组进行打乱
    ori_data_list = [os.listdir(os.path.join(dataset_root,dataset_label[i])) for i in range(6)]
    for i in range(6):
        path = os.path.join(dataset_root,dataset_label[i])
        for j,v in enumerate(ori_data_list[i]):
            ori_data_list[i][j] = os.path.join(path,v)
        random.shuffle(ori_data_list[i])

    # 分组信息保存在列表
    train_list_group_ori = [[] for i in range(val_k)] #目前第0组保存0，真实应该保存1234
    val_list_group = [[] for i in range(val_k)]
    # 数据增强与数据集制作
    print("图像增强中：")
    for i in range(6):
        for j,v in tqdm(enumerate(ori_data_list[i])):
            group =j%val_k
            # dataset_group_root

            Image_enhance(v,i,train_list_group_ori[group],val_list_group[group])


    # 调整训练集 #目前第2组保存2，真实应该保存0134
    train_list_group = [[] for i in range(val_k)]
    for group in range(val_k-1):
        for group2 in range(val_k-1):
            if not group2 == group:
                train_list_group[group].extend(train_list_group_ori[group2])

    for group in range(val_k):
        random.shuffle(train_list_group[group])
    # 写入txt
    for group in range(val_k):
        temp_train_txt = train_txt_prefix + str(group)+ ".txt"
        temp_val_txt = val_txt_prefix + str(group) + ".txt"
        with open(temp_train_txt,'w',encoding='UTF-8') as f:
            f.writelines(train_list_group[group])
            # for i,v in enumerate(train_list_group[group]) :
            #     f.write(v+"\n")
        with open(temp_val_txt,'w',encoding='UTF-8') as f:
            f.writelines(val_list_group[group])
            # for i,v in enumerate(val_list_group[group]) :
            #     f.write(v+"\n")




    # 对于每个类别，分别做以下处理
    # 1.随机打乱  random.shuffle(ori_data[0])
    # 2.对于 ori_data[0]中的每一个数据，它所在的组为其下标%5
    # 3.对于 ori_data[0]中的每一个数据，将其转写到对应组的val_data_group0.txt文件中（包含路径和标签）
    # 4.对于 ori_data[0]中的每一个数据，将其进行数据增强，增强的同时，把增强后的[路径,标签]，转写到一个新的列表train_data = [5个组]中，之后记得再打乱





