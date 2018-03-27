import tensorflow as tf

'''
Training data
'''
x_data = [1, 2, 3]
y_data = [1, 2, 3]

'''
Linear Regression Modeling
'''
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')
hypothesis = X * W

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent Algorithm
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
new_W = W.assign(descent)

'''
Run
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(new_W, feed_dict={X:x_data, Y:y_data})
    print('step :', step, end=', ')
    print('cost :', sess.run(cost, feed_dict={X:x_data, Y:y_data}), end=', ')
    print('weight', sess.run(W))

