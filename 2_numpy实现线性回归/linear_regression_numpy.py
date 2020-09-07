'''
利用numpy实现回归，这个案例很无聊
'''
import numpy as np

data=np.genfromtxt("data.csv", delimiter=",")
# y=w*x+b

# 参数设置
initial_w=0
initial_b=0
epoch=1000
lr=0.0001

def compute_error(w,b,data):
    x=data[:,0].copy()
    y=data[:,1].copy()
    y_pred=w*x+b
    loss=((y-y_pred)**2).mean()
    return loss

def step_gradient(current_w,current_b,data,lr):
    x=data[:,0].copy()
    y=data[:,1].copy()

    y_pred=current_w*x+current_b
    w_gradient=(2*(y_pred-y)*x).mean() # grad_w = 2(wx+b-y)*x
    b_gradient=(2*(y_pred-y)).mean() # grad_b = 2(wx+b-y)

    new_w=current_w-lr*w_gradient
    new_b=current_b-lr*b_gradient
    return new_w,new_b

w=initial_w
b=initial_b
for e in range(epoch):
    w,b=step_gradient(w,b,data,lr)
print('After {0} iterations b = {1}, w = {2}, error = {3}'.format(epoch, b, w, compute_error(w,b,data)))




