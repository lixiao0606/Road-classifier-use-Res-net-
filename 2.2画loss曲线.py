# -*-coding:utf-8-*-
from matplotlib import pyplot as plt
import numpy as np


def ReadData(data_loc):
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []

    # open(data_loc,"r").readlines()
    with open(data_loc, "r") as f:
        linedata = f.readlines()

        for line_i in linedata:
            data = line_i.split('\t')
            print("data = ", data)
            epoch_i , train_loss_i,test_loss_i,test_accuracy_i =data[1], data[3],data[5],data[7]
            epoch_list.append(int(epoch_i))
            train_loss_list.append(float(train_loss_i))
            test_loss_list.append(float(test_loss_i))
            test_accuracy_list.append(float(test_accuracy_i))

    # print(epoch_list)
    # print(train_loss_list)
    # print(test_loss_list)
    # print(test_accuracy_list)
    return epoch_list, train_loss_list  ,test_loss_list,test_accuracy_list



def Draw_Train_Loss(train_loss_list,train_loss_list_2=False):
    plt.style.use('dark_background')

    plt.title("train_loss")
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    # train_loss_list = train_loss_list[:50]
    epoch_list = [i for i in range(len(train_loss_list))]

    p1, = plt.plot(epoch_list, train_loss_list, linewidth=3)
    if train_loss_list_2 == False:
        plt.legend([p1], ["Loss"])
    else:
        p2, = plt.plot(epoch_list, train_loss_list_2, linewidth=3)
        plt.legend([p1, p2], ["Loss1", "Loss2"])

    plt.show()




def Draw_Val_Acc(val_acc_list,val_acc_list_2 = False):
    plt.style.use('dark_background')
    plt.title("test_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("test_accuracy")

    # train_loss_list = train_loss_list[:50]
    epoch_list = [i for i in range(len(val_acc_list))]

    p1, = plt.plot(epoch_list, val_acc_list, linewidth=3)
    if val_acc_list_2 == False:
        plt.legend([p1], ["Acc"])
    else:
        p2, = plt.plot(epoch_list, val_acc_list_2, linewidth=3)
        plt.legend([p1, p2], ["Acc1", "Acc2"])

    plt.show()



if __name__ == '__main__':
    data_1_loc = "output/resnet18_z01.txt"
    # data_2_loc = "output/resnet34_e02.txt"

    _, train_loss_list  ,test_loss_list,test_accuracy_list = ReadData(data_1_loc)
    # _, train_loss_list_2  ,test_loss_list_2,test_accuracy_list_2 = ReadData(data_2_loc)

    Draw_Train_Loss(train_loss_list)

    Draw_Val_Acc(test_accuracy_list)
