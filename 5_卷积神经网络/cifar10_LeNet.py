import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers

# 超参数设置
tf.random.set_seed(78)
num_classes=10
batch_size=64
lr=0.0001
epoch=5

# 数据载入
(x_train,y_train), (x_test,y_test) =keras.datasets.cifar10.load_data()

# 数据预处理函数
def perprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.0
    y=tf.cast(y,dtype=tf.int32)
    y=tf.squeeze(y,axis=1)
    y=tf.one_hot(y,depth=num_classes)
    return x,y

# 设置数据加载器
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
# 设置repeat其实就是相当于对数据集进行epoch次训练，repeat次数设置为epoch是等效的
train_dataloader=train_dataloader.shuffle(50000).batch(batch_size).map(perprocess).repeat(epoch)
test_dataloader=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataloader=test_dataloader.shuffle(10000).batch(batch_size).map(perprocess)

# 模型构建及设置优化器 Lenet
# model=keras.Sequential([
#     keras.layers.Conv2D(6,kernel_size=3,padding='same',strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.ReLU(),
#     keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
#     # 16,16,6
#     keras.layers.Conv2D(16, kernel_size=3, padding='same', strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.ReLU(),
#     keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
#     # 8,8,16
#     keras.layers.Flatten(), # 展平操作
#     keras.layers.Dense(120,input_shape=[8*8*16]),
#     keras.layers.ReLU(),
#     keras.layers.Dense(84),
#     keras.layers.ReLU(),
#     keras.layers.Dense(num_classes)
# ])

# VGG16
model=keras.Sequential([ # 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Flatten(),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(num_classes, activation=None),
])

optimizer=keras.optimizers.Adam(lr=lr)
criterion=keras.losses.CategoricalCrossentropy(from_logits=True)

summary_writer=tf.summary.create_file_writer('./runs')

# 模型训练
loss_metric=keras.metrics.Mean()
loss_metric.reset_states()
for step,(batchx,batchy) in enumerate(train_dataloader):
    with tf.GradientTape() as tape:
        out=model(batchx)
        loss=criterion(batchy,out) # 注意真实值在前，预测值在后
    grads = tape.gradient(loss, model.trainable_variables)  # 对指定param求取梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 优化器更新梯度
    loss_metric.update_state(loss)

    if step%100==0:
        with summary_writer.as_default():
            tf.summary.scalar('train loss',loss_metric.result().numpy(),step)
        print('step:{},train_loss:{}'.format(step,loss_metric.result().numpy()))
        loss_metric.reset_states()

    if step%500==0:
        # 测试
        total_correct=0
        for batchx_,batchy_ in test_dataloader:
            out_=model(batchx_)
            pred=tf.argmax(out_,axis=1)

            y=tf.argmax(batchy_,axis=1) # 注意次数batchy_是独热编码后的数据，要还原到原始标签
            correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct=tf.reduce_sum(correct).numpy()
            total_correct+=correct
        accuracy=total_correct/x_test.shape[0]
        print('step:{},accuracy:{}'.format(step,accuracy))

        with summary_writer.as_default():
            tf.summary.scalar('test accuracy',accuracy,step)











