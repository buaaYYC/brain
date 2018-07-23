from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution3D,Reshape,MaxPooling3D,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
import nibabel as nib
import numpy as np
import pdb
# from zky_load_onehot_data import  data_save,one_hot,down_sample
from zky_GPU_load_data import load_label,load_xdata,onehot_dic
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
    用keras框架来构建的3D神经网络，这个网络共有4层，
    输入：灰质体积图、白质体积图、原始的T1加权MRI图，（都是在mini152空间）
    最终输出的是测试的loss和accuracy
    X_trian,y_train是训练集
    x_test,y_test是验证集
'''
#===========================
#提取训练集和测试集
file_path_x = '/DATA/239/nmzuo/Cam-CAN/processed/'
file_path_y = '/DATA/239/nmzuo/yyc/zky_age_predict/y_data/file.csv'
# x_dic = load_xdata(file_path_x)
y_dic = load_label(file_path_y)
#降采样4倍，保存在train.npy，test.npy
X_train = np.load("/DATA/239/nmzuo/yyc/zky_age_predict/gpu_temp/train.npy")/255
X_test = np.load("/DATA/239/nmzuo/yyc/zky_age_predict/gpu_temp/test.npy")/255
# X_train = x_dic['X_train']/255
# X_test =  x_dic['X_test']/255
#2018/7/20  添加年龄归一化
y_train = y_dic['y_train']
y_test = y_dic['y_test']
######################
#m表示样本数，z,w,h分别表示3D图像的长宽高，channel=1
print("raw_x_test.shape:"+str(X_test.shape))
print("raw_y_test.shape:"+str(y_test.shape))
print("raw_x_train.shape:"+str(X_train.shape))
print("raw_y_train.shape:"+str(y_train.shape))
m,z,w,h = X_train.shape
n = X_test.shape[0]
X_train = X_train.reshape(m,h,z,w,1)
#y_train = y_train.reshape(m,1)
X_test = X_test.reshape(n,h,z,w,1)

######################
'''
onehot_data = onehot_dic(file_path_y)
#onehotdata是储存好年龄的71中类别，18-88岁
#然后我们通过key将对应的onehot值提出，重新组成一个（n，71）的onehot矩阵
y_test_ = []
y_train_ = []
for y1 in y_test:
    y1 = onehot_data[y1]
    y_test_.append(y1)
y_test = np.array(y_test_)
for y2 in y_train:
    y2 = onehot_data[y2]
    y_train_.append(y2)
y_train = np.array(y_train_)
# y_test = dic[20].reshape(1,5)
'''
y_test = y_test.reshape(n,1)
y_train = y_train.reshape(m,1)
print("x_test.shape:"+str(X_test.shape))
print("y_test.shape:"+str(y_test.shape))
print("x_train.shape:"+str(X_train.shape))
print("y_train.shape:"+str(y_train.shape))


dim_y = y_test.shape[1]
######################
"""开始网络的搭建"""
#建立模型
model = Sequential()
print(X_train.shape)

#建立第一层卷积神经网络C1
'''输入图像形状input_shape'''
c = 1
model.add(Convolution3D(
    input_shape=(h,z,w,c),
    filters=5,
    kernel_size=(3,3,3),
    strides=(1,1,1)
))
model.add(Activation('relu'))

#建立第二层卷积神经网络C2
model.add(Convolution3D(
    filters=1,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
))
#建立batch—Normalization层
model.add(BatchNormalization())
#加一个激活函数
model.add(Activation('relu'))

#最大池化层Maxpooling,P3层
model.add(MaxPooling3D(
    pool_size=(2,2,2),
    strides=2,
))
#model.add(Dropout(0.3, noise_shape=None))
#建立一个全连接层，F4层，它生成回归模型以输出大脑预测的年龄。
model.add(Flatten())
model.add(Dense(dim_y))
#dim_y表示label的维数
# 开始训练并且测试
model.compile(
    loss='mean_squared_error',
#'mean_absolute_error',
    optimizer=SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True),
#'rmsprop',
    # 'adam',
#SGD(lr=0.01, decay=0.03, momentum=0.9, nesterov=True),
    metrics=['accuracy']
              )
    #batch_size：对样本进行分类，每组28个，epoch：训练次数,batch_size=28q
model.fit(X_train,y_train,epochs=2000,batch_size=28,shuffle=True)
"""网络搭建结束"""
######################

Y_test_pred = model.predict(X_test)

#计算出 loss和accuracy
loss,accuracy = model.evaluate(X_test,y_test)
print('test loss:',loss)
print('test accuracy:',accuracy)

######################
"""保存模型"""
#将模型序列化为JSON
model_json = model.to_json()
with open("model.json",'w') as json_file:
    json_file.write(model_json)

#将权重保存在HDF5文件下
model.save_weights("model.h5")
print('Save mode to disk')
##########################
