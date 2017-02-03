import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# -1 ~ 1 까지 임의의 값을 할당
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) = Wx + b - Our Hypothesis
hypothesis = W * X + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Black box .. GradientDescentOptimizer - cost 최소화
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수 초기화
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

# 가지고 있던 model 을 이용해서 .. 결과 예측
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
