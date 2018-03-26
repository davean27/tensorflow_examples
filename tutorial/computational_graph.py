import tensorflow as tf

'''
build graph
'''
const_node1 = tf.constant(3.0, tf.float32)
const_node2 = tf.constant(4.0, tf.float32)
com_node = tf.add(const_node1, const_node2) # or const_node1 + const_node2


'''
run graph
'''
sess = tf.Session()

print("sess.run(const_node1, const_node2): ", sess.run([const_node1, const_node2]))
print("sess.run(com_node): ", sess.run(com_node))
