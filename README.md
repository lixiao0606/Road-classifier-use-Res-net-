# Road-classifier-use-Res-net-

## Introduce:
Use resnet to classify road surfaces into the following 6 categories: Asphalt, Cobblestone, Gravel, Icy, Rainy, Snowy.
(使用resnet将路面分为以下6类：沥青、鹅卵石、砾石、结冰、雨天、雪地。)

## How to place the dataset:
Current path has a folder named "dataset".
(当前路径有一个dataset文件夹)
There are 6 sub folders in the "dataset" folder, namely Asphalt, Coblestone, Gravel, Icy, Rainy, Snowy.
(dataset文件夹内有6个子文件夹，分别为Asphalt, Cobblestone, Gravel, Icy, Rainy, Snowy)

## How to train the model
First, run code 1.1 to enhance the data and group the dataset (using k-level cross validation)
(首先运行代码1.1来增强数据以及为数据集分组（使用k级交叉验证）)
Then run code 1.2 to calculate the mean and variance.
(然后运行代码1.2来计算均值和方差)
Then modify the calculated mean and variance in the utils.py file.
(然后用计算得到的均值和方差在utils.py文件中修改)
Finally, run train.py to start training the model.
(最后运行train.py即可开始训练模型)

## Evaluation of the model
First, run code 3.1 to test a single image.
(首先运行代码3.1可以测试单张图片)
Then run code 3.2 to batch test multiple images and save the results in a table file
(然后运行代码3.2可以批量测试多张图片并将结果保存在表格文件中)
Finally, run code 3.3 to calculate the accuracy, recall rate, F1 score, confusion matrix and other evaluation indicators.
(最后运行代码3.3可以计算精度、召回率、F1-score、混淆矩阵等评估指标)
