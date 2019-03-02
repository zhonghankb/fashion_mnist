import tensorflow as tf
import numpy as np
import cv2
import os

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    
    #定义placeholder，用于训练时存放训练样本的输入和输出
    
    # n_H0 ,n_W0 图像的height 和width, n_C0 图像的channels
    x = tf.placeholder(name ="x_input", shape = (None, n_H0, n_W0, n_C0), dtype = tf.float32)
    # n_y0 样本的标签数量
    y = tf.placeholder(name ="y_label", shape = (None, n_y), dtype = tf.float32)
    
    return x, y



def cnn_block1(x,is_training):
    
    # conv2d
    x = tf.layers.conv2d(x, 16,[3,3],padding='same')
    # batch normalization
    x = tf.layers.batch_normalization(x, training=is_training)
    # relu activation
    x = tf.nn.swish(x)
    # maxpooling
    x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)  
    
    
    
    return x

def cnn_block2(x,is_training):
    
    # conv2d
    x = tf.layers.conv2d(x, 32, [3,3],padding='same')
    # batch normalization
    x = tf.layers.batch_normalization(x, training=is_training)
    # relu activation
    x = tf.nn.swish(x)
    # maxpooling
    x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)   
  
    
    return x    
 

    
def cnn_block3(x,is_training):
    
    # conv2d
    x = tf.layers.conv2d(x, 64, [3,3],padding='same')
    # batch normalization
    x = tf.layers.batch_normalization(x,training=is_training)
    # relu activation
    x = tf.nn.swish(x)
    # maxpooling
    x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)   
 
    
    return x

def cnn_block4(x,is_training):
    
    # conv2d
    x = tf.layers.conv2d(x, 128, [3,3],padding='same')
    # batch normalization
    x = tf.layers.batch_normalization(x,training=is_training)
    # relu activation
    x = tf.nn.swish(x)
    # maxpooling
    x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)   
    
 
    
    return x

def cnn_block5(x,is_training):
    
    # conv2d
    x = tf.layers.conv2d(x, 256, [3,3],padding='same')
    # batch normalization
    x = tf.layers.batch_normalization(x,training=is_training)
    # relu activation
    x = tf.nn.swish(x)
    # maxpooling
    x = tf.layers.max_pooling2d(x, pool_size = 2, strides = 2)   
    
    
    return x



def Fashion_CNN(x,is_training):
     
    print("Net_start:",x.get_shape())
    # block2
    x = cnn_block2(x,is_training)
    print("block2:",x.get_shape())
    # block3
    x = cnn_block3(x,is_training)
    print("block3:",x.get_shape())
    # block4
    x = cnn_block4(x,is_training)
    print("block4:",x.get_shape())
    # flattn layer
    x = tf.layers.flatten(x)
    print("flatten:",x.get_shape())
    # fully connected layer     
    x = tf.layers.dense(x,units=128)
    print("fully_connected1:",x.get_shape())
    # batch_normalization
    x = tf.layers.batch_normalization(x,training=is_training)
    # relu
    x = tf.nn.swish(x)
    # fully connected layer        
    x = tf.layers.dense(x,units=10)
    print("fully_connected3:",x.get_shape())
    # softmax layer
    output = tf.nn.softmax(x, name ="prediction")
    print("output",output.get_shape())
    return output



def compute_cost(output,y):
    
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = y, logits = output))
    
    return cost

    
def random_mini_batches(x,y,mini_batch_size,seed):
    
    # 将数据集按照mini_batch_size分割为一定数量的batch，分批进行训练
    m = x.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation,:,:,:]
    shuffled_y = y[permutation,:]
    
    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k*mini_batch_size:(k+1)*mini_batch_size,:,:,:]
        mini_batch_y = shuffled_y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
        
    #当数据集的样本数量不能被mini_batch_size整除时，将最后余下的样本单独作为一个batch    
    if m % mini_batch_size !=0:
        mini_batch_x = shuffled_x[num_complete_minibatches*mini_batch_size:m,:,:,:]
        mini_batch_y = shuffled_y[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    

    
    