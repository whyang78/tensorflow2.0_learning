import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras
from pokemon import load_pokemon

# 超参数设置
tf.random.set_seed(78)
num_classes=5
data_path='./pokemon'
batch_size=64
lr=0.001
epochs=20

#预处理函数
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x,mean=img_mean,std=img_std):
    x=(x-mean)/std
    return x

def preprocess(x,y):
    '''
    :param x: image path
    :param y: label
    :return:
    '''
    # 包含数据增强
    x=tf.io.read_file(x)
    x=tf.image.decode_jpeg(x,channels=3)
    x=tf.image.resize(x,[244,244])

    x=tf.image.random_flip_left_right(x)
    x=tf.image.random_crop(x,[224,224,3])

    x=tf.cast(x,dtype=tf.float32)/255.0
    x=normalize(x)
    y=tf.convert_to_tensor(y)
    y=tf.one_hot(y,depth=num_classes)
    return x,y

# 数据读取
train_images,train_labels,name2label=load_pokemon(data_path,'train')
train_dataloader=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_dataloader=train_dataloader.shuffle(len(train_images)).map(preprocess).batch(batch_size)

val_images,val_labels,name2label=load_pokemon(data_path,'val')
val_dataloader=tf.data.Dataset.from_tensor_slices((val_images,val_labels))
val_dataloader=val_dataloader.map(preprocess).batch(batch_size)

test_images,test_labels,name2label=load_pokemon(data_path,'test')
test_dataloader=tf.data.Dataset.from_tensor_slices((test_images,test_labels))
test_dataloader=test_dataloader.map(preprocess).batch(batch_size)

# 网络搭建
# 加载带有预训练参数的DenseNet121网络模型，去掉最后一层全连接层，并且最后一个池化层设置为Global max pooling
net=keras.applications.DenseNet121(weights='imagenet',include_top=False,pooling='max')
net.trainable=False # 特征提取网络参数冻结
model=keras.Sequential([
    net,
    keras.layers.Dense(1024),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes)
])

# 模型装配
model.compile(optimizer=keras.optimizers.Adam(lr=lr),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 模型训练
# 设置早停
earlystop_callbacks=keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                min_delta=0.001,
                                                patience=5)

model.fit(train_dataloader,validation_data=val_dataloader, validation_freq=1, epochs=epochs,
           callbacks=[earlystop_callbacks])

# 模型测试
model.evaluate(test_dataloader)