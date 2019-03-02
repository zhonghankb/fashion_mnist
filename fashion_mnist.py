import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import matplotlib.pyplot as plt

from Fashion_CNN import *
from data_process import *


(x_data,y_data) = load_data('fashion-mnist_train.csv',False)
(x_data_test,y_data_test) = load_data('fashion-mnist_test.csv',True)

print('Shape of x_data='+str(x_data.shape))
print('Shape of y_data='+str(y_data.shape))
print('Shape of x_data_test='+str(x_data_test.shape))
print('Shape of y_data_test='+str(y_data_test.shape))


y_data = y2indicator(y_data)#训练样本转换为one-hot样式
print('Shape of y data_one_hot='+str(y_data.shape))
y_data_test = y2indicator(y_data_test)

print('y_data[0]='+str(y_data[0]))
print('y_data_test[0]='+str(y_data_test[0]))

#图像水平翻转，扩充训练样本数量
x_data_aug,y_data_aug = image_flip(x_data,y_data)


# 将水平翻转得到的dataset 分割为train和validation两部分，采用5-fold交叉验证，训练集占80%，验证集占20%
ratio_start = 0.0
(x_train,y_train,x_val,y_val) = split_data(x_data,y_data,x_data_aug,y_data_aug, ratio_start)


(num_images,n_H,n_W,n_C) = x_train.shape
print('Shape of train data='+str(x_train.shape))
n_y = y_train.shape[1]
print('n_y='+str(n_y))

costs = []
train_acc = []
val_acc = []

learning_rate = 0.0005
num_epochs = 30
minibatch_size = 256

print_cost = True
seed = 1


x,y = create_placeholders(n_H, n_W, n_C, n_y)
is_training = tf.placeholder(tf.bool)
output = Fashion_CNN(x,is_training)
cost = compute_cost(output, y)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
predict = tf.argmax(output,1)
target = tf.argmax(y,1)
correct_prediction = tf.cast(tf.equal(predict, target),dtype = tf.float32)
accuracy_num = tf.reduce_sum(correct_prediction)

#统计模型的参数数量
total_parameters = 0
for variable in tf.trainable_variables():
    variable_parameters = 1
    for dim in variable.get_shape():
        variable_parameters *= dim.value
    total_parameters += variable_parameters

print("Total number of trainable parameters: %d" % total_parameters)


if x_train.shape[0]%minibatch_size == 0:
    eval_num_batch_train = int(x_train.shape[0]/minibatch_size)
else:
    eval_num_batch_train = int(x_train.shape[0]/minibatch_size)+1
print('eval_num_batch_train='+str(eval_num_batch_train))
if x_val.shape[0]%minibatch_size == 0:
    eval_num_batch_val = int(x_val.shape[0]/minibatch_size)
else:
    eval_num_batch_val = int(x_val.shape[0]/minibatch_size)+1
print('eval_num_batch_val='+str(eval_num_batch_val))
if x_data_test.shape[0]%minibatch_size == 0:
    eval_num_batch_test = int(x_data_test.shape[0]/minibatch_size)
else:
    eval_num_batch_test = int(x_data_test.shape[0]/minibatch_size)+1

eval_minibatches_train = random_mini_batches(x_train, y_train, minibatch_size, seed)
eval_minibatches_val = random_mini_batches(x_val, y_val, minibatch_size, seed)
eval_minibatches_test = random_mini_batches(x_data_test, y_data_test, minibatch_size, seed)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
   
    #训练模型
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(num_images/minibatch_size)+1
        seed = seed + 1
        minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)
        
        for minibatch in minibatches: 
            (minibatch_x, minibatch_y) = minibatch
            _,temp_cost = sess.run([optimizer, cost], feed_dict ={x:minibatch_x, y:minibatch_y,is_training:True})
            minibatch_cost +=temp_cost/num_minibatches
            
        if print_cost == True and epoch%1 ==0:
            print("Cost after epoch %i: %f" %(epoch, minibatch_cost))
            
            eval_accu_train = 0
            for minibatch in eval_minibatches_train:
                (minibatch_x, minibatch_y) = minibatch
                eval_accu_batch = accuracy_num.eval({x:minibatch_x,y:minibatch_y,is_training:False}) 
                eval_accu_train += eval_accu_batch
            train_accuracy =eval_accu_train/(x_train.shape[0])
            print("train_accuracy = %f" %train_accuracy)
            
            eval_accu_val = 0
            for minibatch in eval_minibatches_val:
                (minibatch_x, minibatch_y) = minibatch
                eval_accu_batch = accuracy_num.eval({x:minibatch_x,y:minibatch_y,is_training:False}) 
                eval_accu_val += eval_accu_batch
            val_accuracy =eval_accu_val/(x_val.shape[0])           
            print("val_accuracy = %f" %val_accuracy)
            costs.append(minibatch_cost)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
                
    plt.plot(np.squeeze(costs),color='green', label='cost')
    plt.plot(np.squeeze(train_acc),color='red', label='train_acc')
    plt.plot(np.squeeze(val_acc),color='blue', label='val_acc')
    plt.ylabel("cost")
    plt.xlabel("Iterations per 1")
    plt.title("Learning_rate =" +str(learning_rate)+"BatchSize="+str(minibatch_size)+"Num_Epochs="+str(num_epochs))
    plt.show()


    
    eval_accu_test = 0
    for minibatch in eval_minibatches_test:
            
        (minibatch_x, minibatch_y) = minibatch
        eval_accu_batch = accuracy_num.eval({x:minibatch_x,y:minibatch_y,is_training:False}) 
        eval_accu_test += eval_accu_batch

    test_accuracy =eval_accu_test/(x_data_test.shape[0])
    print("test_accuracy = %f" %test_accuracy) 

    
    

    
    


