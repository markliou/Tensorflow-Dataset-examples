import numpy as np
import tensorflow as tf

data_s = tf.placeholder(tf.float32, [None,4,5])
dataset = tf.data.Dataset.from_tensor_slices(data_s)
dataset = dataset.batch(2) # This can set the batch size. "Batch 1" means "by element". 
# dataset = dataset.repeat() # unstoppable repeating
dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
print(dataset)

# prefetch
dataset = dataset.prefetch(buffer_size=100)

dataset_iter = dataset.make_initializable_iterator()
dataset_go = dataset_iter.get_next()
print(dataset_go)


s = np.reshape([i for i in range(10 * 4 * 5)], [10,4,5])
with tf.Session() as sess:
    sess.run(dataset_iter.initializer, feed_dict={data_s:s})
    for i in range(10): # if the dataset use the repeat(), the dataset can be fetch over and over again.
        print(sess.run(dataset_go))
    

