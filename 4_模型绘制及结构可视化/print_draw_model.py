import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras


# 目前好像Sequential的网络适用
model=keras.Sequential([
    # 注意一定要在第一层网络添加input_shape，其除去batch剩下的维度输入即可
    # 例如mnist输入[b,784]，此处只填[784]即可
    keras.layers.Dense(512,input_shape=[784],activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(10)
])

model.build(input_shape=(1,784))
print(model.summary())

from tensorflow.keras.utils import plot_model
import pydotplus

#参数 ：模型名称，结构图保存位置，是否展示shape
plot_model(model,to_file='model.png',show_shapes=True)





