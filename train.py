from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from unet import UNetp
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from tqdm import tqdm
import torch.optim as optim
from fldataset import FlorisLesDataset

batch_size=4
LES_TRAIN_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/les-yaw/train-sbs-4'
LES_TEST_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/les-yaw/test-sbs-4'
FLORIS_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/floris-yaw'
LOAD_DIR = '/data/add_disk1/dczhang/0Final/weight/yaw-sbs-4'
SAVE_DIR = '/data/add_disk1/dczhang/0Final/weight/variable-yaw-angle'
fl_train = FlorisLesDataset(LES_TRAIN_DIR,FLORIS_DIR)
fl_test = FlorisLesDataset(LES_TEST_DIR,FLORIS_DIR,augment=False)
target= 'LES'
train_dataloader = DataLoader(fl_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(fl_test, batch_size=batch_size, shuffle=False)

model = UNetp(142,142,5,32,batchnorm=False,dropout=0.1).cuda()

model.load_state_dict(torch.load(os.path.join(LOAD_DIR,'30000.pth')), strict= False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epoch_loss = []
pbar = tqdm(range(60000),dynamic_ncols=True)
min_eval_loss = 1e4
train_loss_cum = []
eval_loss_cum = []
for epoch in pbar:  # loop over the dataset multiple times
    model = model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['x'].cuda().float()
        targets = data['y'].cuda().float()
        c = data['c'].cuda().float()
        
        ##change target
        #targets = targets*c[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, c)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()**0.5
    avg_loss = running_loss/(1+i)
    if epoch%2000 == 0:
        model = model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for j, data in enumerate(test_dataloader, 0):
                inputs = data['x'].cuda().float()
                targets = data['y'].cuda().float()
                c = data['c'].cuda().float()
                
                ##change target
                #targets = targets*c[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                outputs = model(inputs, c)
                loss = criterion(outputs, targets)
                eval_loss += loss.item()**0.5
        eval_loss /= (j+1)
        eval_loss_cum.append(eval_loss)
        train_loss_cum.append(avg_loss)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        
        np.save(os.path.join(SAVE_DIR,'train-loss.npy'),np.array(train_loss_cum))
        np.save(os.path.join(SAVE_DIR,'eval-loss.npy'),np.array(eval_loss_cum))
        print(train_loss_cum)
        print(eval_loss_cum)
        if eval_loss < min_eval_loss:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,'%d.pth'%epoch))
            min_eval_loss = eval_loss
    pbar.set_description('%s avg loss: %.5f eval loss: %.5f min eval loss %.5f'%(target, avg_loss, eval_loss, min_eval_loss))
    running_loss = 0.0