import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

print(a)
print(b)

c = a+b

print(c)

print(sess.run(c))
