import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())

