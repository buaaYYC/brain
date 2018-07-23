import numpy as np
import csv
import scipy
import nibabel as nib
from os import listdir
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from skimage import transform
file_path = '/DATA/239/nmzuo/Cam-CAN/processed/'

def load_xdata(file_path,dim_n=4):
    filebox = listdir(file_path)#搜索出路径下的所有文件名
    filebox.sort()
    file_list = []
    train_list = []
    test_list =[]
    X_dic ={}
    i = 0
    for file_name in filebox:
        path = file_path + file_name + "/anat/cat12Atlas/mri/"
        files = listdir(path)
        if i%10==0:#每10个取一个样本来作为测试集
            for file in files:
                if 'mwp1' in file:
                    file_list.append(file)#查找出我们需要的数据集，并将其文件名存入file_list中
                    print("=====runing====:" + str(i))
                    data = nib.load(path + file)  # 通过文件名，导入数据集
                    data_arr = data.get_fdata()
                    data_trans = translateit(data_arr, (10,10,10))#平移
                    data_rotate = rotateit(data_arr, 40)#旋转

                    data_new_arr =down_sample(data_arr,dim_n)#降采样
                    data_new_trans = down_sample(data_trans,dim_n)
                    data_new_rotate = down_sample(data_rotate, dim_n)
                    # print(type(data_arr))
                    test_list.append(data_new_arr)  # 将数据集存入datalist中
                    test_list.append(data_new_trans)
                    test_list.append(data_new_rotate)
        else:
            for file in files:
                if 'mwp1' in file:
                    file_list.append(file)#查找出我们需要的数据集，并将其文件名存入file_list中
                   # print(len(file_list))
                    data = nib.load(path + file)  # 通过文件名，导入数据集
                    data_arr = data.get_fdata()
                    data_trans = translateit(data_arr, (10,10,10))#平移像素
                    data_rotate = rotateit(data_arr, 40)#旋转

                    data_new_arr =down_sample(data_arr,dim_n)#降采样
                    data_new_trans = down_sample(data_trans,dim_n)
                    data_new_rotate = down_sample(data_rotate, dim_n)
                    # print(type(data_arr))
                    train_list.append(data_new_arr)  # 将数据集存入datalist中
                    train_list.append(data_new_trans)
                    train_list.append(data_new_rotate)
        i = i + 1
    test_list_arr = np.array(test_list)
    train_list_arr = np.array(train_list)
    np.save("test.npy", test_list_arr)
    np.save("train.npy",train_list_arr)
    X_dic['X_train'] = train_list_arr
    X_dic['X_test'] = test_list_arr
    return X_dic#返回整合好的数据集,分别是测试集和训练集
def load_label(file_y_path):#输入保存成csv的标签文件
    test_row_list = []
    train_row_list = []
    with open(file_y_path,'r') as fr:
        rows = csv.reader(fr)
        rows = list(rows)[1:]
        i=0
        for row in rows:
            # print(row[1])
            label_row = float(row[1])/88#除以88，就是对年龄进行了归一化
            if (i%10)==0:
                # print(row[0])
                print("is runing :"+str(i))

                test_row_list.append(label_row)
                test_row_list.append(label_row)
                test_row_list.append(label_row)
            else:
                train_row_list.append(label_row)
                train_row_list.append(label_row)
                train_row_list.append(label_row)
            i = i + 1
    test_row_list.sort()
    train_row_list.sort()
    y_dic = {}
    y_dic['y_train'] = np.array(train_row_list)
    y_dic['y_test'] = np.array(test_row_list)
    return y_dic#返回划分好的测试和训练的标签（年龄）

def down_sample(data1,n):#降采样,输入data1是要进行采样的样本，n表示将的倍数
    z, w, h = data1.shape
    data_new = transform.resize(data1,(z // n, w // n, h // n))
    # data_new = np.zeros((z // n, w // n, h // n))
    # z_new, w_new, h_new = data_new.shape
    # for i in range(z_new):
    #    for j in range(w_new):
    #        for k in range(h_new):
    #            data_new[i, j, k] = data1[n * i, n * j, n * k]
    return data_new#反回降采样后的数据
def one_hot(y):
    y_list = list(np.squeeze(y))
    y_dlist = list(set(y_list))#去重
    y_dlist.sort(key=y_list.index)
    y_d = LabelEncoder().fit_transform(y_dlist)
    y_onehot = OneHotEncoder(sparse=False).fit_transform(y_d.reshape(-1, 1))#onehot转换
    dic = {}
    for i in range(len(y_dlist)):
        key = y_dlist[i]
        value = y_onehot[i]
        dic[key] = value
    return  y_onehot,dic#返回one-hot处理后的矩阵，和存储onehot和原始矩阵的字典

def onehot_dic(filepath):#filepath='file.csv'，将全部的y值，转换为onehot，并保存到字典中方便引用
    row_list = []
    with open(filepath,'r') as fr:
        rows = csv.reader(fr)
        i=0
        for row in rows:
            row_list.append(row[1])
    row_list.sort()
    one_,dict = one_hot(np.array(row_list[1:]))
    return dict
"""2018/7/18 更新，为了增加数据而设置的"""
#=================================================================
def translateit(image, offset, isseg=False):#平移像素
    order = 0 if isseg == True else 5

    trans_data = scipy.ndimage.interpolation.shift(image,
                (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')
    return trans_data
def rotateit(image, theta, isseg=False):#图像旋转
    order = 0 if isseg == True else 5
    rotate_data = scipy.ndimage.rotate(image, float(theta),
                                       reshape=False, order=order, mode='nearest')
    return rotate_data
#=================================================================
if __name__=="__main__":
    dic = load_label('file.csv')
    y_test = dic['y_test']
    print(y_test)

