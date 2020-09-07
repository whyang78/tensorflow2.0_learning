import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras

# 超参数设置
tf.random.set_seed(78)
num_classes=10
batch_size=64
lr=0.001
epoch=5

# 数据载入
(x,y), (x_test,y_test) =keras.datasets.cifar10.load_data()

# 数据预处理函数
def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.0
    y=tf.cast(y,dtype=tf.int32)
    y=tf.squeeze(y,axis=1)
    y=tf.one_hot(y,depth=num_classes)
    return x,y

# 样本划分
idx=tf.range(x.shape[0])
idx=tf.random.shuffle(idx)
train_size=int(len(idx)*0.8)
val_size=int(len(idx)*0.2)
test_size=x_test.shape[0]
x_train,y_train=tf.gather(x,idx[:train_size]),tf.gather(y,idx[:train_size])
x_val,y_val=tf.gather(x,idx[train_size:]),tf.gather(y,idx[train_size:])

# 设置数据加载器
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataloader=train_dataloader.shuffle(train_size).batch(batch_size).map(preprocess)
val_dataloader=tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataloader=val_dataloader.shuffle(val_size).batch(batch_size).map(preprocess)
test_dataloader=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataloader=test_dataloader.batch(batch_size).map(preprocess)

# 网络结构
class BasicBlock(keras.layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock, self).__init__()
        self.left=keras.Sequential([
            keras.layers.Conv2D(filter_num,kernel_size=3,strides=stride,padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filter_num,kernel_size=3,strides=1,padding='same'),
            keras.layers.BatchNormalization()
        ])

        if stride!=1:
            self.short_cut=keras.layers.Conv2D(filter_num,kernel_size=1,strides=stride,padding='same')
        else:
            self.short_cut=lambda x:x

    def call(self,inputs,training=None):
        output=self.left(inputs)
        identify=self.short_cut(inputs)
        output=keras.layers.add([output,identify])
        output=tf.nn.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self,layers_num,num_classes=10):
        super(ResNet, self).__init__()
        self.conv=keras.Sequential([keras.layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same'),
                                keras.layers.BatchNormalization(),
                                keras.layers.Activation('relu'),
                                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        self.layer1=self._make_layer(64,layers_num[0])
        self.layer2=self._make_layer(128,layers_num[1],2)
        self.layer3=self._make_layer(256,layers_num[2],2)
        self.layer4=self._make_layer(512,layers_num[3],2)

        self.pool=keras.layers.GlobalAveragePooling2D()
        self.fc=keras.layers.Dense(num_classes)

    def call(self,inputs,training=None):
        out=self.conv(inputs)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)

        out=self.pool(out)
        out=self.fc(out)
        return out

    def _make_layer(self,filter_num,block_num,stride=1):
        Block=keras.Sequential()
        Block.add(BasicBlock(filter_num,stride))

        for i in range(1,block_num):
            Block.add(BasicBlock(filter_num,1))
        return Block
resnet18=ResNet([2,2,2,2])

# 模型装配
resnet18.compile(optimizer=keras.optimizers.Adam(lr=lr),
                 loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# 模型训练
history=resnet18.fit(train_dataloader,epochs=epoch,validation_data=val_dataloader,validation_freq=1)

# 模型预测
result=resnet18.evaluate(test_dataloader)

keras.applications.resnet










