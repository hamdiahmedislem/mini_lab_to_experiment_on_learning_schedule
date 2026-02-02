import numpy as np
from functions import bias_weight_loader , bias_weight_saver
import os
from PIL import Image
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr
from numba import njit
# Setting
if __name__ == "__main__" :
    load_default = True
    module_num = 3
else :
    load_default = False
    module_num = 2
size_batch_default = 25
number_of_samples_over_two = 5000
number_of_iterations_default = 1000000
lr_default = 0.001
####################
def load(dir_path : str) ->np.array:
    file_path_in_dir = os.listdir(dir_path)
    img_loaded = []
    i = 0
    for file_path in file_path_in_dir :
        print(i)
        img = Image.open(dir_path+file_path)
        img_np = np.array(img).reshape(-1,1)/255
        img_loaded.append(img_np)
        i += 1
    return np.array(img_loaded)

empty_img_tr  = load("e:/lolol/python/ML_T_empty,non_empty/DATA_0/empty/")
nempty_img_tr = load("e:/lolol/python/ML_T_empty,non_empty/DATA_0/notempty/")
empty_img_ev  = load("e:/lolol/python/ML_T_empty,non_empty/DATA_0/empty_evaluate/")
nempty_img_ev = load("e:/lolol/python/ML_T_empty,non_empty/DATA_0/notempty_evaluate/")
print(True)

def Mix_arrays(first_em : np.array, second_nem : np.array , start : int , size : int) :
    f_data = np.empty((size,625))
    f_data[0::2] = first_em[start:(size//2) + (size%2) + start:1].reshape(-1,625)
    f_data[1::2] = second_nem[start:size//2+ start:1].reshape(-1,625)
    pre = np.empty(size)
    pre[0::2] = 0
    pre[1::2] = 1
    return f_data , pre

#####################
class linear_layer(object) :
    def __init__(self,number_of_inputs : int,number_of_neurons : int,number_of_lay : int) -> None:
        if not load_default :
            self.bias = np.zeros((1,number_of_neurons))
            self.weight = np.random.rand(number_of_inputs,number_of_neurons)
        else :
            self.bias , self.weight = bias_weight_loader(module_num,number_of_lay) 
        #print(f"layer.bias{number_of_lay} : {self.bias.shape}, layer.wheight{number_of_lay} : {self.wheight.shape}, ")
# it is advisable to enter number_of_lay_mod as num of layer directly after the number of the module
    def forward(self,inputs : np.array) -> np.array:

        self.output = np.dot(inputs,self.weight) + self.bias
        return self.output

def RELU(X_RELU : np.array) -> np.array:
    return np.maximum(X_RELU,0)

def SIGMOID(X_SIGMOID : np.array) -> np.array:
    return 1/ (1 + np.exp(-X_SIGMOID))

def BCE(Y_BCE_pred,Y_BCE_true):
    Y_BCE_pred = np.clip(Y_BCE_pred, 1e-9 , 1-1e-9)
    return np.mean(-((1-Y_BCE_true)*np.log(1-Y_BCE_pred)+Y_BCE_true*np.log(Y_BCE_pred)))

# Module 0

layer_1 = linear_layer(625,5,1)
layer_2 = linear_layer(5,1,2)

def Module_0(X_M_0 : np.array) -> np.array:

    X_M_0 = layer_1.forward(X_M_0)
    A2 = RELU(X_M_0)
    X_M_0 = layer_2.forward(A2)
    X_M_0 = SIGMOID(X_M_0)

    return X_M_0,A2

def train(lr : float,size_batch : int , number_of_iterations : int,
          exp : str, update_lr):
    global empty_img_tr , nempty_img_tr
    plot = dict(batch_list = [], loss_list = [], acc_list = [], lr_list = []
                ,cv = True , batch_num = number_of_iterations
                ,batch_s = size_batch , l_acc = 0 , b_rea = number_of_iterations)
    v1 , v2 , acc = 0 , 0 , 0
    if update_lr :
        lr_0 = lr
        var1 , var2 , var3 , var11 , var12 , var13 = symbols("x y z a b c")
        expr = parse_expr(exp)

    for i in range(number_of_iterations) :
        X , Y = Mix_arrays(empty_img_tr,nempty_img_tr,((i+1)*size_batch)%number_of_samples_over_two-size_batch 
                           if ((i+1)*size_batch)%number_of_samples_over_two > size_batch else 
                           0,size_batch)
        Y = Y.reshape(-1,1)
        Y_pred,a2 = Module_0(X)

        if i == 0 :
            loss_t = BCE(Y_pred,Y)
        else :
            loss_t = loss_t_1
        loss_t_1 = BCE(Y_pred,Y)
        if update_lr :
            lr = float(expr.subs({var1:lr,var2:loss_t,var3:loss_t_1,
                                        var11:lr_0 , var12:i , var13:size_batch}).evalf())
        print(f"batch {i} | loss(t1) : {loss_t_1:.7f} | lr : {lr:.7f} | acc : {acc}")
        if i % 100 == 0 :
            acc = evaluate()
            plot["batch_list"].append(i) ; plot["loss_list"].append(loss_t_1)
            plot["acc_list"].append(acc) ; plot["lr_list"].append(lr)
            bias_weight_saver(layer_1.bias ,layer_1.weight ,module_num,1)
            bias_weight_saver(layer_2.bias ,layer_2.weight ,module_num,2)
            plot["l_acc"] = acc
            if acc >= 1 :
                plot["b_rea"] = i
                break
            if lr >= 250 or lr < 1e-6 :
                plot["cv"] = False
                break
        def backpropogation() :
            nonlocal v1 , v2
            #calculate dy 
            dy_dz2 = Y_pred - Y
            dy_dz1 = np.dot(dy_dz2.reshape(size_batch,1),layer_2.weight.T) * (a2 > 0)
            #calculate dw1 , dw2 ; dz1 , dz2
            dw2 = np.dot(a2.T,dy_dz2)
            dw2 = dw2.reshape(-1,1)
            dw1 = np.dot(X.T,dy_dz1)/len(X)
            dz1 = np.mean(dy_dz1,axis=0)
            dz2 = np.mean(dy_dz2)
            #v1 and v2 updating
            v1 =v1*0.1 + dw1
            v2 =v2*0.1 + dw2
            #debuging #print(f"layer_1 : {layer_1.wheight.shape} layer_2 : {layer_2.wheight.shape} dw1 : {dw1.shape} dw2 : {dw2.shape} | v1 : {v1.shape} v2 : {v2.shape}")
            layer_2.weight -= lr*v2
            layer_1.weight -= lr*v1
            layer_1.bias -= lr*dz1
            layer_2.bias -= lr*dz2
        backpropogation( )
    return plot

def evaluate() :
    global empty_img_ev , nempty_img_ev
    X , Y = Mix_arrays(empty_img_ev,nempty_img_ev,0,1000)
    Y = Y.reshape(-1,1)
    Y_pred,a2 = Module_0(X)
    acc = np.mean(Y == (Y_pred >0.5))
    return acc

def back_prop_old(X,ai,Y,at) :
    at = at.reshape(-1)
    P = at - Y
    start = True
    for i in range(len(layer_1.weight[0])) :
        if start :
            P_W = np.dot(P,layer_2.weight[i][0])
            start = False
        else :
            P_W = np.append(P_W,np.dot(P,layer_2.weight[i][0]))
    P_W = P_W.reshape(5,25)
    print(np.mean(X,axis=0))
    delta = lr_default*np.dot(layer_1.weight,np.mean(P_W))
    mean_to_delta = np.mean(X,axis=0)
    for j in range(len(layer_1.weight)) :
        delta[j] = delta[j]*mean_to_delta
    layer_1.weight = layer_1.weight - lr_default*np.mean(X,axis=0)*np.dot(layer_1.weight,np.mean(P_W))
    for i in range(len(P)) :
        ai[i] = np.dot(P[i],ai[i])
    ai = np.mean(ai, axis=0)
    layer_2.weight = layer_2.weight - ai

def main(lr : float = lr_default ,
         size_batch : int = size_batch_default ,
         number_of_iterations : int = number_of_iterations_default ,
         exp : str = "",
         update_lr : bool = False) -> None :
    return train(lr,size_batch,number_of_iterations,
          exp=exp,update_lr=update_lr)

if __name__ == "__main__" :
    main()