import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras

# 超参数设置
tf.random.set_seed(78)
batch_size=64
epoch=5
lr=0.01
use_regularization=True # 是否加入正则化损失
c=0.001

# 数据集载入
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

# 预处理函数
def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.0
    x=tf.reshape(x,[-1,28*28])
    y=tf.cast(y,dtype=tf.int32)
    y=tf.one_hot(y,depth=10)
    return x,y

# 构建数据加载器
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataloader=train_dataloader.shuffle(x_train.shape[0]).batch(batch_size).map(preprocess).repeat(epoch)
test_dataloader=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataloader=test_dataloader.shuffle(x_test.shape[0]).batch(batch_size).map(preprocess)

# 搭建网络
# 自定义网络层
class MyDense(keras.layers.Layer):
    def __init__(self,in_dim,out_dim):
        super(MyDense, self).__init__()
        self.kernel=self.add_weight('w',[in_dim,out_dim],trainable=True)
        self.bias=self.add_weight('b',[out_dim],trainable=True)

    def call(self,inputs,training=None):
        # 全连接层的运算与网络状态（训练或者测试）无关，故此处可以不用管training
        out=inputs@self.kernel + self.bias
        return out

# 自定义网络
class mlp(keras.Model):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = MyDense(28 * 28, 512)
        self.fc2 = MyDense(512, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self,inputs,training=None):
        x=self.fc1(inputs)
        x=tf.nn.relu(x)
        x=self.fc2(x)
        x = tf.nn.relu(x)
        x=self.fc3(x)
        x = tf.nn.relu(x)
        x=self.fc4(x)
        x=tf.nn.dropout(x,0.5) # 加入了dropout,注意设置training
        x = tf.nn.relu(x)
        out=self.fc5(x)
        return out

model=mlp()

# 设置损失函数、优化器、测量器
criterion=keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer=keras.optimizers.Adam(lr=lr)
accuracy_metric=keras.metrics.CategoricalAccuracy()
loss_metric=keras.metrics.Mean()

# 设置可视化 使用tensorboard记录模型训练过程
summary_writer=tf.summary.create_file_writer('./logs')

# 模型训练
loss_metric.reset_states()
for step,(batchx,batchy) in enumerate(train_dataloader):
    with tf.GradientTape() as tape:
        out=model(batchx,training=True)
        loss=criterion(batchy,out)

        if use_regularization:
            # l2正则化损失
            loss_l2_regularization=[]
            for p in model.trainable_variables:
                loss_l2_regularization.append(tf.sqrt(tf.nn.l2_loss(p)))
            loss=loss+ c*tf.reduce_sum(tf.stack(loss_l2_regularization))
        loss_metric.update_state(loss)

    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))

    if step%100==0:
        with summary_writer.as_default():
            tf.summary.scalar('train loss',loss_metric.result().numpy(),step=step)
        print('step:{},train_loss:{}'.format(step,loss_metric.result().numpy()))
        loss_metric.reset_states()

    if step%500==0:
        accuracy_metric.reset_states()
        correct=0
        for batchx_,batchy_ in test_dataloader:
            out=model(batchx_,training=False)
            pred=tf.argmax(out,1)

            y_true=tf.argmax(batchy_,1)
            correct+=tf.reduce_sum(tf.cast(tf.equal(y_true,pred),dtype=tf.int32)).numpy()
            accuracy_metric.update_state(batchy_,out)
        accuracy=correct/x_test.shape[0]
        print('step:{},test_accuracy:{},metric_accuracy:{}'.format(step,accuracy,accuracy_metric.result().numpy()))

        with summary_writer.as_default():
            tf.summary.scalar('test accuracy',accuracy_metric.result().numpy(),step=step)

