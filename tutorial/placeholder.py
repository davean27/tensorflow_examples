import tensorflow as tf

'''
build graph
'''
num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)
out = num1 + num2 # shortcut for tf.add(num1, num2)

'''
run graph
'''
sess = tf.Session()
print(sess.run(out, feed_dict={num1:1.5, num2:2.5}))
print(sess.run(out, feed_dict={num1:[1.2, 1.3], num2:[4.5, 5.7]}))
