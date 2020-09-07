import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

tf.random.set_seed(78)

# 超参数设置
num_classes=10
batch_size=64
lr=0.001
epoch=5

# 载入数据，此处仅使用trainset进行训练
(x_train,y_train), _ =keras.datasets.mnist.load_data() # 返回两个元祖，第一个是训练集数据及标签，第二个是验证集的
print(x_train.shape,y_train.shape) # [60000,28,28] [60000] 注意此处数据是numpy类型

# 数据预处理
x_train=tf.convert_to_tensor(x_train,dtype=tf.float32)/255.0 # 转换成tensor类型并进行归一化[0,1]
y_train=tf.convert_to_tensor(y_train,dtype=tf.int32)
y_train=tf.one_hot(y_train,depth=num_classes)# 将标签转化成tensor类型后并进行独热编码
print(x_train.shape,y_train.shape) # [60000,28,28] [60000,10]

# 制作加载器
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataloader=train_dataloader.batch(batch_size)

# 模型构建及设置优化器
model=keras.Sequential([
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(num_classes)
])
optimizer=keras.optimizers.SGD(learning_rate=lr)

# 模型训练
losses=[]
for e in range(epoch):
    total_loss=0.0
    for i,(batchx,batchy) in enumerate(train_dataloader):
        with tf.GradientTape() as tape: # 梯度监控
            batchx=tf.reshape(batchx,(-1,28*28)) # [b,28,28]->[b,28*28]
            out=model(batchx)
            # 此处使用MSE，分类问题一般不使用MSE
            loss=tf.reduce_sum(tf.square(out-batchy))/batchx.shape[0]

        grads=tape.gradient(loss,model.trainable_variables) # 对指定param求取梯度
        optimizer.apply_gradients(zip(grads,model.trainable_variables)) # 优化器更新梯度
        total_loss+=loss.numpy()*batchx.shape[0]

    avarage_loss=total_loss/x_train.shape[0] # 每个epoch的平均损失
    print('epoch:',e+1,'  loss:',avarage_loss)
    losses.append(avarage_loss)

# 绘制损失变化
plt.plot(range(1,len(losses)+1),losses)
plt.xticks(range(1,len(losses)+1))
plt.show()



