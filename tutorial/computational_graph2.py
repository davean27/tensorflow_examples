import tensorflow as tf

'''
build graph
'''
const_node1 = tf.constant(3.0, tf.float32)
const_node2 = tf.constant(4.0, tf.float32)
const_node3 = tf.constant(1.2, tf.float32)
com_node1 = tf.add(const_node1, const_node2)
com_node2 = tf.add(com_node1, const_node3)


'''
run graph
'''
sess = tf.Session()

print("sess.run(const_node1, const_node2): ", sess.run([const_node1, const_node2]))
print("sess.run(com_node1): ", sess.run(com_node1))
print("sess.run(com_node2): ", sess.run(com_node2))
