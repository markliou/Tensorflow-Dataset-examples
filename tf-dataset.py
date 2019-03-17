import numpy as np
import tensorflow as tf

data_s = np.reshape([i for i in range(3 * 4 * 5)], [3,4,5])
print(data_s)

dataset = tf.data.Dataset.from_tensor_slices(data_s)
print(dataset)

dataset_iter = dataset.make_initializable_iterator()
dataset_go = dataset_iter.get_next()

with tf.Session() as sess:
    sess.run(dataset_iter.initializer)
    print(sess.run(dataset_go))
    print(sess.run(dataset_go))


