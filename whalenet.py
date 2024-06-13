import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile 
import torch
from torchaudio.transforms import Resample
if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
import os
import numpy as np
from kymatio.torch import Scattering1D

file = ZipFile('/home/alicciardi/whale_marine_mammals/data.zip', "r")
results_tot = {}
def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


test=[]
for item in file.namelist():
    if '_' not in item and item!= 'data/' and 'wav' in item:
        #print(item)
        pos=findOccurrences(item,'/')
        test.append(item[pos[0]+1:pos[1]])
classes_set=list(set(test))

y_labs = []
data_full_list=[]
srate_full_list=[]
count = 0
for item in file.namelist():
        if '_' not in item and item!= 'data/' and 'wav' in item:
        
            pos=findOccurrences(item,'/')
            name_class=item[pos[0]+1:pos[1]]
            file_op=file.open(item,'r')

            mf = torchaudio.info(file_op)
            file_op=file.open(item,'r')
            if mf.bits_per_sample in [16,24,32]:
        
                x, sr = torchaudio.load(file_op)
                res=Resample(orig_freq=sr,new_freq=47600)
                data_full_list.append(res(x))
                y_labs.append(name_class)
                srate_full_list.append(sr)
            count += 1
            if count%500 ==0:
                 print(count)

signal_len = np.array([x.shape[1]/srate_full_list[k] for (k,x) in enumerate(data_full_list)])
avg = np.mean(signal_len)
sd = np.std(signal_len)
idx = np.where(signal_len <= avg+100*sd)[0]
selected_items = signal_len[idx]
from torch.nn.functional import pad
def cutter(X,cut_point): #cuts and centers
    cut_list = []
    cut_point = int(cut_point)
    j = 0

    for x in X:
        n_len = x.shape[1]
        add_pts = cut_point-n_len

        if (n_len<= cut_point):
            pp_left = int(add_pts/2)
            pp_right = add_pts - pp_left
            cut_list.append(pad(x, (pp_left,pp_right)))

        else :

            center_time = int(n_len/2)
            pp_left = int(cut_point-center_time)
            pp_right = cut_point - pp_left
            cut_list.append(x[:,center_time-pp_left: center_time+ pp_right])
        j += 1

    return torch.cat(cut_list)

y_sel = np.array(y_labs)[idx]
data_sel = [data_full_list[j] for j in idx]

lens = [x.shape[1] for x in data_sel]
cut_point= 8000
X_cut = cutter(data_sel, cut_point)
dict = {}

for name in set(y_labs):
    dict[name] = np.where(y_sel == name)

keep_idx = []
thrs = 50

for k in dict.keys():
    els = len(dict[k][0])
    if els > thrs:
        keep_idx.append(dict[k][0])

keep = set()
#[keep.add(set(l)) for l in keep_idx]

for l in keep_idx:
    keep = keep.union(set(l))
kp = list(keep)
XC = X_cut[kp, :]
#XP = X_pad[kp,:]
y = y_sel[kp]


def standardize(X):
    st = torch.std(X, dim=1, keepdim=True)
    mn = torch.mean(X, dim=1, keepdim=True)
    return (X - mn) / st


X = standardize(XC)
X = XC

df_X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])

# Add a column for y
df_X['y'] = y
df_X_no_duplicates = df_X.drop_duplicates(subset=df_X.iloc[:, 0:cut_point])
X = torch.from_numpy(df_X_no_duplicates.iloc[:, 0:8000].values)
y = df_X_no_duplicates.iloc[:, -1].values

batches = [64, 128, 256]

from torchaudio.transforms import MelSpectrogram
batch_size = 500
N_batch =  int(X.shape[0]/batch_size)+1
spectr = MelSpectrogram(normalized= True, n_mels = 64).to(device)
mx = []
for n in range(N_batch):
    x = X[n*batch_size:(n+1)*batch_size].to(device)
    MX_tmp = spectr(x)
    mx.append(MX_tmp)
MX = torch.concat(mx)
MX = MX.unsqueeze(1)

batches = [64, 128, 256]
JQ = [(7, 10), (6, 16), (8,14)]


J, Q = JQ[2]
T = X.shape[1]

scattering=Scattering1D(J,T,Q)
scattering.cuda()
wst = []
for n in range(N_batch):
    x = X[n*batch_size:(n+1)*batch_size].to(device)
    SX_tmp = scattering(x)
    wst.append(SX_tmp)
SX = torch.concat(wst)


print(SX.shape)


meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)


def median_norm(X):
    md = torch.median(X)
    sn = torch.std(X)
    return (X - md) / sn





SX_med = SX
for i in range(SX.shape[0]):
    SX_med[i][order0] = median_norm(SX[i][order0])
    SX_med[i][order1] = median_norm(SX[i][order1])
    SX_med[i][order2] = median_norm(SX[i][order2])
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

batch_size = 256

from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
y_trc = torch.as_tensor(lbe.fit_transform(y))
index_shuffle=np.arange(len(y_trc))
idx_train, idx_test, y_trXX, y_testXX = train_test_split(index_shuffle, y_trc, test_size=.25, stratify=y)

batch_size = batches[0]

train_dataset_mel = TensorDataset(MX.cpu()[idx_train], y_trXX)
val_dataset_mel = TensorDataset(MX.cpu()[idx_test], y_testXX)
train_dataloader_mel = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=True)
val_dataloader_mel = DataLoader(val_dataset_mel, batch_size=batch_size, shuffle=False)

train_dataset_1 = TensorDataset(SX_med[idx_train][:,order1].cpu(), y_trXX)
val_dataset_1 = TensorDataset(SX_med[idx_test][:,order1].cpu(), y_testXX)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False)
train_dataset_2 = TensorDataset(SX_med[idx_train][:,order2].cpu(), y_trXX)
val_dataset_2 = TensorDataset(SX_med[idx_test][:,order2].cpu(), y_testXX)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
val_dataloader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)

import torch.nn as nn

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Instantiate the model
model_mel = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
model_wst_1=ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
model_wst_2=ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
learning_rate_mel = .01
optimizer_mel = torch.optim.AdamW(model_mel.parameters(), lr=learning_rate_mel,amsgrad= True, weight_decay= .001 )
scheduler_mel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mel, 'min')
learning_rate_1 = .01
optimizer_1 = torch.optim.AdamW(model_wst_1.parameters(), lr=learning_rate_1,amsgrad= True, weight_decay= .001 )
scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min')
learning_rate_2 = .01
optimizer_2 = torch.optim.AdamW(model_wst_2.parameters(), lr=learning_rate_2,amsgrad= True, weight_decay= .001 )
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min')

def training_resnet(model,train_dataloader,val_dataloader,learning_rate,optimizer,scheduler, fname):
    criterion = nn.CrossEntropyLoss()
    
    n_total_steps = len(train_dataloader)
    num_epochs = 100
    loss_train = []
    acc_train = []
    acc_eval = []
    loss_eval = []
    for epoch in range(num_epochs):
    
        loss_ep_train = 0
        n_samples = 0
        n_correct = 0
        for i, (x, labels) in enumerate(train_dataloader):
    
            x = x.to(device)
    
            labels = labels.to(device, dtype=torch.long)
    
            #forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
    
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            loss_ep_train += loss.item()
            _, predictions = torch.max(outputs, 1)
    
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
    
    
    
            if (i + 1) % 100 == 0:
                print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ')
    
        acc_tr = 100 * n_correct / n_samples
        acc_train.append(acc_tr)
        loss_train.append(loss_ep_train/len(train_dataloader))
    
        loss_ep_eval = 0
    
        with torch.no_grad():
    
            n_correct = 0
            n_samples = 0
    
            for x, labels in val_dataloader:
                x = x.to(device)
    
                labels = labels.to(device)
                outputs = model(x)
                lossvv = criterion(outputs, labels)
    
                _, predictions = torch.max(outputs, 1)
    
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()
                loss_ep_eval += lossvv.item()
    
            acc = 100 * n_correct / n_samples
    
        acc_eval.append(acc)
        loss_eval.append(loss_ep_eval/len(val_dataloader))
    
        print(f' validation accuracy = {acc}')
    
    res = np.array([loss_train, loss_eval, acc_train, acc_eval])
    
    
    namefile = f'{fname}_{J,Q}_{batch_size}'
    np.save(namefile, res)

    from sklearn.metrics import roc_auc_score
    yp = []
    ytr = []
    y_prob = []
    import time 
    times=[]
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
    
        for x, labels in val_dataloader:
            x = x.to(device)
    
            labels = labels.to(device)
            tick=time.time()
            outputs = model(x)
            tick=time.time()-tick
            times.append(tick/len(labels))
            pr_out = torch.softmax(outputs, dim = 1)
    
            proba, predictions = torch.max(pr_out, 1)
    
            yp.append(predictions.cpu().numpy())
            ytr.append(labels.cpu().numpy())
            y_prob.append(pr_out.cpu().numpy())
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
    
        
        acc = 100 * n_correct / n_samples


training_resnet(model_mel,train_dataloader_mel,val_dataloader_mel,learning_rate_mel,optimizer_mel,scheduler_mel, 'modelmel')
training_resnet(model_wst_1,train_dataloader_1,val_dataloader_1,learning_rate_1,optimizer_1,scheduler_1,'modelws1')
training_resnet(model_wst_2,train_dataloader_2,val_dataloader_2,learning_rate_2,optimizer_2,scheduler_2, 'modelws2')

train_dataloader_1_fin = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=False)
val_dataloader_1_fin = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False)
train_dataloader_2_fin = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=False)
val_dataloader_2_fin = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)
list_prob_1=[]
list_prob_2=[]
list_prob_1_val=[]
list_prob_2_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_1_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_1(x)
        list_prob_1.append(outputs)
    for x, labels in val_dataloader_1_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_1(x)
        list_prob_1_val.append(outputs)
    for x, labels in train_dataloader_2_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_2(x)
        list_prob_2.append(outputs)
    for x, labels in val_dataloader_2_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_2(x)
        list_prob_2_val.append(outputs)


prob_train_1=torch.concat(list_prob_1)
prob_train_2=torch.concat(list_prob_2)
train=torch.hstack((prob_train_1,prob_train_2))
prob_val_1=torch.concat(list_prob_1_val)
prob_val_2=torch.concat(list_prob_2_val)
val=torch.hstack((prob_val_1,prob_val_2))


train_final = TensorDataset(train, y_trXX)
val_final = TensorDataset(val, y_testXX)
train_final_load = DataLoader(train_final, batch_size=batch_size, shuffle=True)
val_final_load = DataLoader(val_final, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear=nn.Linear(64,256)
        self.activation=nn.ReLU()
        self.linear2=nn.Linear(256,128)
        self.activation=nn.ReLU()
        self.linear3=nn.Linear(128,32)

    
        

    def forward(self, x):
        out=self.linear(x)
        out=self.activation(out)
        out=self.linear2(out)
        out=self.activation(out)
        out=self.linear3(out)
        return out

model_MLP = MLP().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = .001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_total_steps = len(train_final_load)
num_epochs = 500
loss_train = []
acc_train = []
acc_eval = []
loss_eval = []
for epoch in range(num_epochs):

    loss_ep_train = 0
    n_samples = 0
    n_correct = 0
    for i, (x, labels) in enumerate(train_final_load):

        x = x.to(device)

        labels = labels.to(device)

        #forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()



        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ')

    acc_tr = 100 * n_correct / n_samples
    acc_train.append(acc_tr)
    loss_train.append(loss_ep_train/len(train_final_load))

    loss_ep_eval = 0

    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for x, labels in val_final_load:
            x = x.to(device)

            labels = labels.to(device)
            outputs = model_MLP(x)
            lossvv = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            loss_ep_eval += lossvv.item()

        acc = 100 * n_correct / n_samples
    acc_eval.append(acc)
    loss_eval.append(loss_ep_eval/len(val_final_load))

    if epoch%100==0:
        print(f' validation accuracy = {acc}')
res = np.array([loss_train, loss_eval, acc_train, acc_eval])

namefile = f'S1+S2_{J,Q}_{batch_size}'
np.save(namefile, res)

    
train_dataloader_mel_fin = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=False)
val_dataloader_mel_fin = DataLoader(val_dataset_mel, batch_size=batch_size, shuffle=False)
list_prob_mel=[]
list_prob_mel_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_mel_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_mel(x)
        list_prob_mel.append(outputs)
    for x, labels in val_dataloader_mel_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_mel(x)
        list_prob_mel_val.append(outputs)


train_dataloader_star = DataLoader(train_final, batch_size=batch_size, shuffle=False)
val_dataloader_star = DataLoader(val_final, batch_size=batch_size, shuffle=False)
list_prob_star=[]
list_prob_star_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_star:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_MLP(x)
        list_prob_star.append(outputs)
    for x, labels in val_dataloader_star:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_MLP(x)
        list_prob_star_val.append(outputs)

prob_train_star=torch.concat(list_prob_star)
prob_train_mel=torch.concat(list_prob_mel)
# train=torch.hstack((prob_train_1,prob_train_2))
prob_val_star=torch.concat(list_prob_star_val)
prob_val_mel=torch.concat(list_prob_mel_val)
# val=torch.hstack((prob_val_1,prob_val_2))

outputs=torch.max(torch.hstack((prob_val_star.unsqueeze(1),prob_val_mel.unsqueeze(1))),1)[0]
print(outputs.shape)
_, predictions = torch.max(outputs, 1)
n_correct = (predictions == y_testXX.cuda() ).sum().item()
accuracy=n_correct/predictions.shape[0]
final_res = {}
print(f'{accuracy}')
final_res['max_merge'] = accuracy
def get_best_lambda(pi_train1, pi_train2, pi_val1, pi_val2, y_train, y_val, n_lambda = 11):

    lambda_range = np.linspace(0,1, n_lambda)

    res = {}

    for l in lambda_range:

        pi_end_train = l * pi_train1 + (1- l) * pi_train2

        pi_end_val = l * pi_val1 + (1- l) * pi_val2

 

        _, pred_train = torch.max(pi_end_train, dim = 1)

        correct_predictions = (pred_train == y_train).sum()

        acc_train = correct_predictions/y_train.shape[0]

 

        _, pred_val = torch.max(pi_end_val, dim = 1)

        correct_predictions = (pred_val == y_val).sum()

        acc_val = correct_predictions/y_val.shape[0]

 

        res[l] = [acc_train.item(), acc_val.item()]
    return res
print(get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.cuda(), y_testXX.cuda(), n_lambda = 31))
final_res['lambdas'] = get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.cuda(), y_testXX.cuda(), n_lambda = 31)
train_hard=torch.hstack((prob_train_star,prob_train_mel))
val_hard=torch.hstack((prob_val_star,prob_val_mel))
train_boh = TensorDataset(train_hard, y_trXX)
val_boh = TensorDataset(val_hard, y_testXX)
train_hard_load = DataLoader(train_boh, batch_size=batch_size, shuffle=True)
val_hard_load = DataLoader(val_boh, batch_size=batch_size, shuffle=False)

model_MLP = MLP().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = .001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_total_steps = len(train_hard_load)
num_epochs = 500
loss_train = []
acc_train = []
acc_eval = []
loss_eval = []
for epoch in range(num_epochs):

    loss_ep_train = 0
    n_samples = 0
    n_correct = 0
    for i, (x, labels) in enumerate(train_hard_load):

        x = x.to(device)

        labels = labels.to(device)

        #forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()



        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ')

    acc_tr = 100 * n_correct / n_samples
    acc_train.append(acc_tr)
    loss_train.append(loss_ep_train/len(train_hard_load))

    loss_ep_eval = 0

    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for x, labels in val_hard_load:
            x = x.to(device)

            labels = labels.to(device)
            outputs = model_MLP(x)
            lossvv = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            loss_ep_eval += lossvv.item()

        acc = 100 * n_correct / n_samples
    acc_eval.append(acc)
    loss_eval.append(loss_ep_eval/len(val_hard_load))

    if epoch%100==0:
        print(f' validation accuracy = {acc}')
    
res = np.array([loss_train, loss_eval, acc_train, acc_eval])

namefile = f'MLP_S+Mel{J,Q}_{batch_size}'
np.save(namefile, res)