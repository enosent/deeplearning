import tensorflow as tf

x_data = [[1, 1, 1, 1, 1], [0., 2., 0., 4., 0.], [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

# -1 ~ 1 까지 임의의 값을 할당
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

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
