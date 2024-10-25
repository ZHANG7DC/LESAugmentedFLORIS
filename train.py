import os
import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNetp
from fldataset import FlorisLesDataset

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_file='train_config.yaml'):
    # Load configuration from YAML
    config = load_config(config_file)

    batch_size = config['batch_size']
    LES_TRAIN_DIR = config['les_train_dir']
    LES_TEST_DIR = config['les_test_dir']
    FLORIS_DIR = config['floris_dir']
    LOAD_DIR = config['load_dir']
    SAVE_DIR = config['save_dir']
    TARGET = config['target']

    INPUT_CHANNELS = config['input_channels']
    OUTPUT_CHANNELS = config['output_channels']
    NUM_CLASSES = config['num_classes']
    HIDDEN_CHANNELS = config['hidden_channels']
    BATCHNORM = config['batchnorm']
    DROPOUT = config['dropout']

    LR = config['learning_rate']
    EPOCHS = config['epochs']
    EVAL_INTERVAL = config['eval_interval']

    # Prepare datasets and dataloaders
    fl_train = FlorisLesDataset(LES_TRAIN_DIR, FLORIS_DIR)
    fl_test = FlorisLesDataset(LES_TEST_DIR, FLORIS_DIR, augment=False)

    train_dataloader = DataLoader(fl_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(fl_test, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = UNetp(INPUT_CHANNELS, OUTPUT_CHANNELS, NUM_CLASSES, HIDDEN_CHANNELS, 
                  batchnorm=BATCHNORM, dropout=DROPOUT).cuda()

    model.load_state_dict(torch.load(os.path.join(LOAD_DIR, '30000.pth')), strict=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop setup
    epoch_loss = []
    min_eval_loss = 1e4
    train_loss_cum = []
    eval_loss_cum = []

    pbar = tqdm(range(EPOCHS), dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            inputs = data['x'].cuda().float()
            targets = data['y'].cuda().float()
            c = data['c'].cuda().float()

            optimizer.zero_grad()
            outputs = model(inputs, c)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() ** 0.5

        avg_loss = running_loss / (i + 1)

        if epoch % EVAL_INTERVAL == 0:
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for j, data in enumerate(test_dataloader, 0):
                    inputs = data['x'].cuda().float()
                    targets = data['y'].cuda().float()
                    c = data['c'].cuda().float()

                    outputs = model(inputs, c)
                    loss = criterion(outputs, targets)
                    eval_loss += loss.item() ** 0.5

            eval_loss /= (j + 1)
            eval_loss_cum.append(eval_loss)
            train_loss_cum.append(avg_loss)

            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            np.save(os.path.join(SAVE_DIR, 'train-loss.npy'), np.array(train_loss_cum))
            np.save(os.path.join(SAVE_DIR, 'eval-loss.npy'), np.array(eval_loss_cum))

            if eval_loss < min_eval_loss:
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{epoch}.pth'))
                min_eval_loss = eval_loss

        pbar.set_description(f'{TARGET} avg loss: {avg_loss:.5f} eval loss: {eval_loss:.5f} min eval loss {min_eval_loss:.5f}')

        running_loss = 0.0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training script for UNetp model.')
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Path to the config YAML file.')
    args = parser.parse_args()

    main(config_file=args.config)
