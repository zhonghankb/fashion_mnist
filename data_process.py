import numpy as np
import pandas as pd

def y2indicator(Y):#将样本label转换为one-hot格式
    N = len(Y)#样本数量
    K = len(set(Y))#去重之后，得到样本label类别数量，
    
    I = np.zeros([N,K])#生成一个NxK的0矩阵，N为样本量，K为label类别数量
    I[np.arange(N), Y] = 1#将每一行中的样本label值对应的列标记为1，从而转换为one-hot形式
    
    return I

def get_label_data(x_data,y_data,label):#将训练样本中特定label的全部样本提取出来
    x_label_data = []
    y_label_data = []
    for i in range(x_data.shape[0]):
        if y_data[i,label]==1:
            #y_label_new = new_label
            x_label_data.append(x_data[i,:,:,:])
            y_label_data.append(y_data[i,:])
    
    x_label_data = np.array(x_label_data)
    y_label_data = np.array(y_label_data)
    
    return x_label_data,y_label_data

def load_data(data_dir,do_shuffle):
    
    data = pd.read_csv(data_dir)
    data = data.values
    if do_shuffle == True:
        np.random.shuffle(data)
    
    x_data = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
    y_data = data[:, 0].astype(np.int32)
    
    return x_data,y_data

def image_flip(x_data,y_data):
    #数据扩充(所有图像水平翻转)
    (data_size,n_H,n_W,n_C) = x_data.shape
    n_y = y_data.shape[1]
    print('original_train_data_size='+str(data_size))

    x_data_aug = np.zeros((data_size,n_H,n_W,n_C),dtype=np.float)
    y_data_aug = np.zeros((data_size,n_y),dtype=np.float)

    for i in range(data_size):
        x_data_aug[i,:,:,0]= np.fliplr(x_data[i,:,:,0])
        y_data_aug[i]= y_data[i]
    
    
    return x_data_aug, y_data_aug

def split_data(x_data,y_data,x_data_aug,y_data_aug,ratio_start):
    
    data_size = x_data.shape[0]
    data_size_aug = x_data_aug.shape[0]
    
    ratio_start = ratio_start
    ratio_end = ratio_start+0.2
    
    # 将原始的train dataset 分割为train和validation两部分
    x_train1 = x_data[0:int(ratio_start*data_size),:,:,:]
    y_train1 = y_data[0:int(ratio_start*data_size),:]

    x_train2 = x_data[int(ratio_end*data_size):data_size,:,:,:]
    y_train2 = y_data[int(ratio_end*data_size):data_size,:]

    x_train = np.concatenate((x_train1,x_train2), axis=0)    
    y_train = np.concatenate((y_train1,y_train2), axis=0) 

    x_val = x_data[int(ratio_start*data_size):int(ratio_end*data_size),:,:,:]
    y_val = y_data[int(ratio_start*data_size):int(ratio_end*data_size),:]
    
    
    #将扩充数据集分割为train 和 validation
    x_train_aug1 = x_data_aug[0:int(ratio_start*data_size_aug),:,:,:]
    y_train_aug1 = y_data_aug[0:int(ratio_start*data_size_aug),:]

    x_train_aug2 = x_data_aug[int(ratio_end*data_size_aug):data_size_aug,:,:,:]
    y_train_aug2 = y_data_aug[int(ratio_end*data_size_aug):data_size_aug,:]

    x_train_aug = np.concatenate((x_train_aug1,x_train_aug2), axis=0) 
    y_train_aug = np.concatenate((y_train_aug1,y_train_aug2), axis=0) 

    x_val_aug = x_data_aug[int(ratio_start*data_size_aug):int(ratio_end*data_size_aug),:,:,:]
    y_val_aug = y_data_aug[int(ratio_start*data_size_aug):int(ratio_end*data_size_aug),:]


    #将分割后的原始数据和扩充数据分别合并，保证训练集和验证集内的原始和扩充数据比重一致
    x_train = np.concatenate((x_train,x_train_aug), axis=0)    
    y_train = np.concatenate((y_train,y_train_aug), axis=0) 

    x_val = np.concatenate((x_val,x_val_aug), axis=0)    
    y_val = np.concatenate((y_val,y_val_aug), axis=0) 
    
    
    return x_train,y_train,x_val,y_val