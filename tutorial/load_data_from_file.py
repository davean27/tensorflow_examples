import tensorflow as tf
import numpy as np

'''
Load from data file
'''
data = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)

# get x_data, y_data from file
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

'''
print data loaded from file
'''
print('x_data shape ', x_data.shape, end=' ')
print('x_data length ', len(x_data))
print('x_data ')
print(x_data)
print('y_data shape ', y_data.shape)
print('y_data ')
print(y_data)

