import tensorflow as tf
import matplotlib.pyplot as plt

'''
Training_data
'''
x_data = [1, 2, 3]
y_data = [1, 2, 3]


'''
Linear Regression Modeling
'''
W = tf.placeholder(tf.float32)
hypothesis = x_data * W  # Simplified hypothesis

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

'''
Run
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_values = []
cost_values = []

for i in range(-30, 50):
    feed_w = i * 0.1
    curr_cost, curr_w = sess.run([cost, W], feed_dict = {W:feed_w})
    w_values.append(curr_w)
    cost_values.append(curr_cost)

plt.plot(w_values, cost_values)
plt.show()

