import numpy as np
from pathlib import Path

commands_ : list[list[int]]
values_ : list[int]

def bias_weight_loader(Num_M:int,Num_L:int) -> np.array :
    try :
        bias : np.array = np.load(f"models/numpy{Num_M}/bias{Num_L}.npy")
        wheight : np.array = np.load(f"models/numpy{Num_M}/wheight{Num_L}.npy")
        return bias , wheight
    finally :
        pass

def bias_weight_saver(bias : np.array,wheight : np.array,Num_M:int,Num_L:int) -> None :

    Path(f"models/numpy{Num_M}").mkdir(parents=True,exist_ok=True)

    np.save(f"models/numpy{Num_M}/bias{Num_L}.npy",bias)
    np.save(f"models/numpy{Num_M}/wheight{Num_L}.npy",wheight)

#######
def do_nothing(val : int,val_ : int):
    return values_[val_-1]
def add(val : int,val_ : int):
    return val + values_[val_-1]
def min(val : int,val_ : int):
    return val - values_[val_-1]
def mul(val : int,val_ : int):
    return val * values_[val_-1]
def dev(val : int,val_ : int):
    return val / values_[val_-1]

math_func = [ do_nothing, add, min, mul,  dev]
def process(commands : list[list[int]],values : list[int]):
    global commands_ , math_func , values_
    commands_ = commands
    values_ = values
    def calcule(start_value : int=0,i : int=1) :
        if len(commands) > i :
            return calcule(math_func[commands_[i-1][1]](start_value,commands_[i][0]),i+1)
        else :
            return start_value
    return calcule()
#######