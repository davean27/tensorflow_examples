import tensorflow as tf

# Constant Node
hello = tf.constant("Hello, TensorFlow!")

# New Session
sess = tf.Session()

# Run
print(sess.run(hello))
