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
epoch=20

# 数据载入
(x_train,y_train), (x_test,y_test) =keras.datasets.mnist.load_data()

# 数据预处理函数
def perprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.0
    x=tf.reshape(x,[-1,28*28])
    y=tf.cast(y,dtype=tf.int32)
    y=tf.one_hot(y,depth=num_classes)
    return x,y

# 设置数据加载器
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
# 设置repeat其实就是相当于对数据集进行epoch次训练，repeat次数设置为epoch是等效的
train_dataloader=train_dataloader.shuffle(60000).batch(batch_size).map(perprocess).repeat(epoch)
test_dataloader=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataloader=test_dataloader.shuffle(10000).batch(batch_size).map(perprocess)

# 模型构建及设置优化器
model=keras.Sequential([
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(num_classes)
])

## 类实现
# class mlp(keras.Model):
#     def __init__(self):
#         super(mlp, self).__init__()
#         self.fc1=keras.layers.Dense(512, activation='relu')
#         self.fc2=keras.layers.Dense(256, activation='relu')
#         self.fc3=keras.layers.Dense(num_classes)
#
#     def call(self, inputs,training=None):  # 训练时设置training=True,验证测试时设置training=False
#         x=self.fc1(inputs)
#         x=self.fc2(x)
#         x=self.fc3(x)
#         return x
# model=mlp()

optimizer=keras.optimizers.SGD(learning_rate=lr)
# criterion=keras.losses.MeanSquaredError() # MSE
criterion=keras.losses.CategoricalCrossentropy(from_logits=True) #交叉熵损失，对于多分类问题，该损失函数收敛更快，效果更好

# 模型训练
train_losses=[]
test_accuracy=[]
for step,(batchx,batchy) in enumerate(train_dataloader):
    with tf.GradientTape() as tape:
        out=model(batchx)
        loss=criterion(batchy,out) # 注意真实值在前，预测值在后
    grads = tape.gradient(loss, model.trainable_variables)  # 对指定param求取梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 优化器更新梯度

    if step%100==0:
        train_losses.append(loss.numpy())

    if step%500==0:
        # 测试
        total_correct=0
        for batchx_,batchy_ in test_dataloader:
            out_=model(batchx_)
            prob=tf.nn.softmax(out_,axis=1)
            pred=tf.argmax(prob,axis=1)

            y=tf.argmax(batchy_,axis=1) # 注意次数batchy_是独热编码后的数据，要还原到原始标签
            correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct=tf.reduce_sum(correct).numpy()
            total_correct+=correct
        accuracy=total_correct/x_test.shape[0]
        test_accuracy.append(accuracy)
        print('step:{},accuracy:{}'.format(step,accuracy))

plt.figure()
plt.plot(train_losses)
plt.show()

plt.figure()
plt.plot(test_accuracy)
plt.show()







