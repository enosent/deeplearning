import tensorflow as tf
import numpy as np

xy = np.loadtxt("train.txt", unpack=True, dtype="float32")
x_data = xy[0:-1]
y_data = xy[-1]

print("x", x_data)
print("Y", y_data)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))

# H(x) = Wx + b - Our Hypothesis
# matmul == matrix multiply
hypothesis = tf.matmul(W, x_data)

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Black box .. GradientDescentOptimizer - cost 최소화
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수 초기화
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
