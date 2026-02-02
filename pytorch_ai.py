import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

num_model = 1
num_bashes = 4
size_bashe = 25
learn_rate = 0.01

em_data = os.listdir("DATA_0/empty")
nem_data = os.listdir("DATA_0/notempty")
PATH = "models/m"+str(num_model)+"para.pt"

def create_x_y(start_idx):
    xs = []
    ys = []
    for j in range(size_bashe):
        ind = start_idx * size_bashe + j
        if ind % 2 == 0:
            img = Image.open("DATA_0/empty/" + em_data[ind // 2])
            label = 1
        else:
            img = Image.open("DATA_0/notempty/" + nem_data[ind // 2])
            label = 0
        x = np.array(img, dtype=np.float32).reshape(-1) / 255.0
        xs.append(torch.from_numpy(x))
        ys.append(label)

    xx = torch.stack(xs)
    y = torch.tensor(ys).float().unsqueeze(1)

    return xx, y


class binary_classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(625,256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(256,1)
    def forward(self , x:torch.tensor) :
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x

loss_fun = nn.BCEWithLogitsLoss()

model_0 = binary_classification()

if os.path.exists(PATH) :
    model_0.load_state_dict(torch.load(PATH, weights_only=False))

optim = torch.optim.SGD(model_0.parameters(),lr=learn_rate)

loss_list = np.array([])
loss_mean_list = np.array([])

for i in range(num_bashes):
    X,y_true = create_x_y(i)

    model_0.train()

    Y = model_0(X)

    loss = loss_fun(Y,y_true.float())
    loss_list = np.append(loss_list,loss.item())

    optim.zero_grad()

    loss.backward()

    optim.step()

    if i % 1 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(Y)
            preds = (probs > 0.5).float()
            acc = (preds == y_true).float().mean()
        loss_mean = loss_list.mean()
        loss_mean_list = np.append(loss_mean_list,loss_mean)
        loss_list = np.array([])
        print(f"batch {i:4d} | loss {loss_mean:.4f} | acc {acc:.3f}")
plt.plot(np.arange(len(loss_mean_list)),loss_mean_list)
#print(loss_mean_list)
#plt.show()

torch.save(model_0.state_dict(),PATH)