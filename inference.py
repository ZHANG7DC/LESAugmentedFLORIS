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
WEIGHT_FILE = '/data/add_disk1/dczhang/0Final/weight/variable-yaw-angle/0.pth'
LES_TEST_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/les-yaw/test'
FLORIS_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/floris-yaw'
SAVE_DIR = '/data/add_disk1/dczhang/0Final/data-postprocessing/les-yaw/variable-yaw-angle/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
fl_test = FlorisLesDataset(LES_TEST_DIR,FLORIS_DIR,augment=False)

test_dataloader = DataLoader(fl_test, batch_size=1, shuffle=False)

model = UNetp(142,142,5,32,batchnorm=False).cuda()

criterion = nn.MSELoss()
model = model.eval().cuda()
model.load_state_dict(torch.load(WEIGHT_FILE))
pbar = tqdm(enumerate(test_dataloader, 0))
running_loss = 0.0
for i, data in pbar:
        # get the inputs; data is a list of [inputs, labels]
    with torch.no_grad():
        inputs = data['x'].cuda().float()
        targets = data['y'].cuda().float()
        c = data['c'].cuda().float()
        fn = data['fn'][0]
        # forward + backward + optimize
        outputs = model(inputs, c)
        loss = criterion(outputs, targets)
    print(fn)
    np.save(os.path.join(SAVE_DIR,fn),outputs[0].detach().cpu().numpy())
        # print statistics
    running_loss += loss.item()**0.5
    avg_loss = running_loss/(i+1)
    pbar.set_description('avg loss: %.5f'%avg_loss)