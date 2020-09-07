import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0' # 选择编号为0的GPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# 超参数设置
tf.random.set_seed(78)
batch_size=64
epoch=5
lr=0.01

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28]) # 注意此处reshape的形状，因为map是在dataloader之后的，其传入单张图像进行处理
    # 之前写的是[-1,28*28]，因为map是在dataloader的batch之后，故传入的是一个batch的图像
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

(x,y),(x_test,y_test)=keras.datasets.mnist.load_data()

idx=tf.range(x.shape[0])
idx=tf.random.shuffle(idx)
train_size=int(len(idx)*0.8) # 训练集数目
val_size=int(len(idx)*0.2) # 验证集数目
test_size=x_test.shape[0] # 测试集数目
x_train,y_train=tf.gather(x,idx[:train_size]),tf.gather(y,idx[:train_size])
x_val,y_val=tf.gather(x,idx[train_size:]),tf.gather(y,idx[train_size:])

# 训练与验证集打乱顺序；测试集不打乱顺序
train_dataloader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataloader=train_dataloader.map(preprocess).shuffle(train_size).batch(batch_size)
val_dataloader=tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataloader=val_dataloader.map(preprocess).shuffle(val_size).batch(batch_size)
test_dataloader=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataloader=test_dataloader.map(preprocess).batch(batch_size)

model=keras.Sequential([
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dropout(0.5),  # 加入dropout，防止过拟合，注意训练时传入参数training=True,测试验证时training=False
    keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(10)
])

# 模型装配
model.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

# 模型训练及验证
# 可以加入早停机制
earlystop_callback=keras.callbacks.EarlyStopping(
  monitor='val_accuracy', min_delta=0.001,
  patience=1)

history=model.fit(train_dataloader,epochs=epoch,callbacks=[earlystop_callback],validation_data=val_dataloader,validation_freq=1)

# 保存模型
model.save_weights('./weights.ckpt') #仅保存权重
#model.save('model.h5') #保存整个模型

# 载入模型
#model=models.load_model('model.h5', compile=False) # 加载整个模型
del model
model = Sequential([keras.layers.Dense(512,activation='relu'),
                    keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dense(64, activation='relu'),
                    # keras.layers.Dropout(0.5),  # 加入dropout，防止过拟合，注意训练时传入参数training=True,测试验证时training=False
                    keras.layers.Dense(32, activation='relu'),
                    # keras.layers.Dropout(0.5),
                    keras.layers.Dense(10)])

model.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
model.load_weights('./weights.ckpt') # 加载权重，故之前要先定义网络结构

# 模型测试
result=model.evaluate(test_dataloader)

# 准确率计数器
acc_meter = metrics.Accuracy()
# acc_meter=keras.metrics.CategoricalAccuracy() # 传入(batchy,pred)即可，不用换成标签
acc_meter.reset_states()
for batchx,batchy in test_dataloader:
    pred=model.predict(batchx) #[b,10]
    y_pred=tf.argmax(pred,1)
    y_pred=tf.cast(y_pred,dtype=tf.int32)

    y_true=tf.argmax(batchy,1)
    y_true=tf.cast(y_true,dtype=tf.int32)
    acc_meter.update_state(y_true,y_pred) # 传入的都是标签，没有独热编码
print(acc_meter.result().numpy())

