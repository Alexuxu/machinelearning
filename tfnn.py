import tensorflow as tf
import numpy as np
import r


class xor():
    def __init__(self, shape):
        self.shape = shape

    def func(self, dt):
        if(dt[0]+dt[1])<0.5:
            rt = [0]
        elif (dt[0]+dt[1])>1.5:
            rt = [0]
        else:
            rt = [1]
        return rt

    def getvalue(self):
        self.value = np.array(list(map(self.func, self.data)))
        return self.value

    def getdata(self):
        self.data = np.random.random(self.shape)
        return self.data


x = tf.placeholder(tf.float32, shape=[None, 2])
y_c = tf.placeholder(tf.float32, shape=[None, 1])

W1=tf.Variable(tf.truncated_normal([2,2],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[2]))
fc1=tf.nn.sigmoid(tf.matmul(x,W1)+b1)

W2=tf.Variable(tf.truncated_normal([2,1],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[1]))
y_=tf.nn.sigmoid(tf.matmul(fc1,W2)+b2)

loss = tf.reduce_mean(tf.square(y_ - y_c))
train = tf.train.AdamOptimizer(1e-2).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



Gendata = xor([50, 2])
Test = xor([500, 2])
tx = Test.getdata()
ty = Test.getvalue()
for i in range(5000):
    x_data = Gendata.getdata()
    y_data = Gendata.getvalue()
    sess.run(train, feed_dict={x: x_data, y_c: y_data})
    if i%1000==0:
        print(sess.run(loss, feed_dict={x:tx, y_c:ty}))

print(sess.run(y_, feed_dict={x: [[0, 0]]}))
print(sess.run(y_, feed_dict={x: [[1, 0]]}))
print(sess.run(y_, feed_dict={x: [[0, 1]]}))
print(sess.run(y_, feed_dict={x: [[1, 1]]}))
