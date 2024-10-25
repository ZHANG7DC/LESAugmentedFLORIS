import os
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNetp
from fldataset import FlorisLesDataset

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_file='inference.yaml'):
    # Load configuration from the YAML file
    config = load_config(config_file)

    WEIGHT_FILE = config['weight_file']
    LES_TEST_DIR = config['les_test_dir']
    FLORIS_DIR = config['floris_dir']
    SAVE_DIR = config['save_dir']
    BATCH_SIZE = config['batch_size']
    INPUT_CHANNELS = config['input_channels']
    OUTPUT_CHANNELS = config['output_channels']
    PARAM_DIM = config['param_dim']

    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Prepare dataset and data loader
    fl_test = FlorisLesDataset(LES_TEST_DIR, FLORIS_DIR, augment=False)
    test_dataloader = DataLoader(fl_test, batch_size=BATCH_SIZE, shuffle=False)

    # Load the model and set it to evaluation mode
    model = UNetp(INPUT_CHANNELS, OUTPUT_CHANNELS, PARAM_DIM).cuda()
    model.load_state_dict(torch.load(WEIGHT_FILE))
    model = model.eval()

    # Set loss criterion
    criterion = nn.MSELoss()

    # Initialize progress bar and tracking variables
    pbar = tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader))
    running_loss = 0.0

    # Inference loop
    for i, data in pbar:
        with torch.no_grad():
            inputs = data['x'].cuda().float()
            targets = data['y'].cuda().float()
            c = data['c'].cuda().float()
            fn = data['fn'][0]

            # Forward pass
            outputs = model(inputs, c)
            loss = criterion(outputs, targets)

            # Save output
            np.save(os.path.join(SAVE_DIR, fn), outputs[0].detach().cpu().numpy())
            print(f'Saved: {fn}')

            # Track loss
            running_loss += loss.item() ** 0.5
            avg_loss = running_loss / (i + 1)
            pbar.set_description(f'avg loss: {avg_loss:.5f}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference with UNetp model.')
    parser.add_argument('--config', type=str, default='inference.yaml', help='Path to the config YAML file.')
    args = parser.parse_args()

    main(config_file=args.config)
