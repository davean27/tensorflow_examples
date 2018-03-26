import tensorflow as tf

# Training Set
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = x * W + b

# reduce_mean : 평균 내주는 함수
cost = tf.reduce_mean(tf.square(h - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = opt.minimize(cost)

# Learning
sess = tf.Session()

# Variable 사용시 반드시 initializer 사용
sess.run(tf.global_variables_initializer())

for step in range(4001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={x: [1.0, 2.0, 3.0, 4.0],
                                                    y: [1.1, 2.1, 3.1, 4.1]})
    if step % 50 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(h, feed_dict={x: [10, 20]}))
