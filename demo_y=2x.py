import tensorflow  as tf
import numpy as np
import matplotlib.pyplot as plt

################ 准备数据 start ##############################
train_x = np.linspace(-1,1,100)
train_y = 2 * train_x + np.random.randn(*train_x.shape)*0.3 # y=2X
plt.plot(train_x,train_y,'ro',label='Original data')
plt.legend()
plt.show()

################ 准备数据 end ##############################

################ 搭建模型 start ##############################
#创建模型
#创建占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
#正向搭建模型
#前向结构
z = tf.multiply(X,W) + b

#反向搭建模型
#反向优化
#定义一个cost,它等于生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))
#定义一个学习率，代表调整参数的速度，一般小于1，越大速度越快，越小调整精度越高
learning_rate = 0.01
optmizer = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(cost)

################ 搭建模型 end ##############################

#迭代训练模型
#初始化所有变量
init = tf.global_variables_initializer()
#定义参数
training_epochs = 20
display_step = 2
#启动 session
with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize":[],"loss":[]}
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,  train_y):
            sess.run(optmizer, feed_dict={X:x,Y:y})
        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print(" Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_x,Y:train_y}),"W=",sess.run(W),"b=",sess.run(b))