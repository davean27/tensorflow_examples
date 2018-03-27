import tensorflow as tf

'''
Build TensorFlow graph
'''
# Input train data set
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# Internal variables (weight, bias)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = x * W + b

# reduce_mean : average value
difference = hypothesis - y
cost = tf.reduce_mean(tf.square(difference))

# get train node
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

'''
Run Tensorflow graph
'''
sess = tf.Session()

# Must initialize global variable before using it
sess.run(tf.global_variables_initializer())


'''
Learning iteration
'''
test_data = {x:[1.0, 2.0, 3.0, 4.0], y:[1.1, 2.1, 3.1, 4.1]}

for step in range(2001):
    cost_val, weight, bias, _ = sess.run([cost, W, b, train], feed_dict=test_data)

    # trace learning step
    if step % 50 == 0:
        print('step :', step, end=', ')
        print('cost_val :', cost_val, end=', ')
        print('weight', weight, end=', ')
        print('bias', bias )


'''
test : prediction
'''
print('Prediction x:10', sess.run(hypothesis, feed_dict=sess.run(hypothesis, feed_dict={x: [10, 20]})))

