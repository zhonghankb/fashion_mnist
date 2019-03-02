# fashion_mnist
simple image classification model by tensorflow

model architecture:  
1:conv(3x3x32)  
2:batch_norm  
3:activation:swish  
4:maxpooling(2,2)  
5:conv(3x3x64)
6:batch_norm
7:activation:swish
8:maxpooling(2,2)
9:conv(3x3x128)
10:batch_norm
11:activation:swish
12:maxpooling(2,2)
13:flattern
14:fully_connected:128
15:batch_norm
16:activation:swish
17:fully_connected:10
18:softmax

num of total parameters = 242250

data_preprocess:
Horizontal flip by 100%

5-fold cross validation

train_accuracy = 0.9805
val_accuracy = 0.9398
test_accuracy = 0.9213
