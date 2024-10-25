from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def cropnpad(dat, x_start, y_start):
    ndat = np.pad(dat, pad_width=((0,0),(50,50),(100,50)),mode='edge')
    return ndat[:,x_start:x_start+280,y_start:y_start+450]

class FlorisLesDataset(Dataset):

    def __init__(self,les_dir, floris_dir, augment=True):
        self.fpath = floris_dir
        self.lpath = les_dir
        self.case = []
        for fn in os.listdir(les_dir):
            if not 'param' in fn:
                self.case.append(fn[:-4])
        self.augment= augment
        print(self.case)
    def __len__(self):
        return len(self.case)

    def __getitem__(self, idx):
        fn = self.case[idx]
        #print(fn[-6:-4])
        
        
        if self.augment:
            x_start = np.random.randint(0,100)
            y_start = np.random.randint(0,150)
            return {'x':cropnpad(np.load(os.path.join(self.fpath,self.case[idx]+'.npy')),x_start,y_start),'y':cropnpad(np.load(os.path.join(self.lpath,self.case[idx]+'.npy')),x_start,y_start),'fn':self.case[idx], 'c':np.load(os.path.join(self.fpath,self.case[idx]+'-param.npy'))}
        else:
            return {'x':np.load(os.path.join(self.fpath, self.case[idx]+'.npy')),'y':np.load(os.path.join(self.lpath,self.case[idx]+'.npy')),'fn':self.case[idx], 'c':np.load(os.path.join(self.fpath,self.case[idx]+'-param.npy'))}