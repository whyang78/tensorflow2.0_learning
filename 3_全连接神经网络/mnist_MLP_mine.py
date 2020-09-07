import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

# 超参数设置
tf.random.set_seed(78)
num_classes=10
batch_size=64
lr=0.001
epoch=15

# 数据集载入
(x_train,y_train),(_)=keras.datasets.mnist.load_data()

# 数据预处理
x_train=tf.convert_to_tensor(x_train,dtype=tf.float32)/255.0
y_train=tf.convert_to_tensor(y_train,dtype=tf.int32)
y_train=tf.one_hot(y_train,depth=num_classes)

# 设置加载器
dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)

# 设置优化器
optimizer=keras.optimizers.SGD(learning_rate=lr)

# 训练参数初始化 有三层网络[b, 784] => [b, 256] => [b, 128] => [b, 10]
w1=tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1=tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
trainable_variables=[w1,b1,w2,b2,w3,b3]

# 训练
losses=[]
for e in range(epoch):
    total_loss = 0.0
    for batchx,batchy in dataloader:
        batchx = tf.reshape(batchx, [-1, 784])
        with tf.GradientTape() as tape:
            h1=batchx@w1+b1
            h1=tf.nn.relu(h1)

            h2=h1@w2+b2
            h2=tf.nn.relu(h2)

            out=h2@w3+b3
            loss=tf.reduce_mean(tf.square(out-batchy))
            total_loss += loss.numpy() * batchx.shape[0]

        # 计算梯度
        grads=tape.gradient(loss,trainable_variables)
        # 优化器更新梯度
        optimizer.apply_gradients(zip(grads,trainable_variables))

    avarage_loss = total_loss / x_train.shape[0]  # 每个epoch的平均损失
    print('epoch:', e + 1, '  loss:', avarage_loss)
    losses.append(avarage_loss)

# 绘制损失变化
plt.plot(range(1,len(losses)+1),losses)
plt.xticks(range(1,len(losses)+1))
plt.show()

