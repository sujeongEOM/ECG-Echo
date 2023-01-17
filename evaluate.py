"""
python evaluate.py resnet
"""

import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d_mse
import numpy as np

if __name__ == "__main__":
    import pandas as pd
    import argparse
    import yaml
    from torch.utils.data import DataLoader
    from warnings import warn
    import torch.nn as nn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('script_yaml',
                        help='script file (yaml) for run')
    cmd_args = parser.parse_args()

    with open(f'{cmd_args.script_yaml}.yaml') as f:
        args = yaml.safe_load(f) #in dictionary

    data = args["data"]
    setup = args["setup"]
    module_model = args["module"]["model"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(data['folder'], module_model['model_name'])

    tqdm.write("Get data...")

    # Get csv data
    test_df = pd.read_csv(data['test']['csv'])
    test_traces = np.load(data['test']['trace'], 'r+')
    n_total = len(test_traces)

    print(f'traces shape: {test_traces.shape}')
    print(f'test df shape: {len(test_df)}')
    
    tqdm.write("Done!")

    tqdm.write("Testing...")
    # Evaluate on test data
    n_total, n_samples, n_leads = test_traces.shape
    n_batches = int(np.ceil(n_total/setup['batch_size']))

    end=0
    tqdm.write("Define model...")
    N_LEADS = 8  # the 8 leads
    N_CLASSES = 1  # just the score

    model = ResNet1d_mse(input_dim=(N_LEADS, setup['seq_length']),
                blocks_dim=list(zip(module_model['net_filter_size'], module_model['net_seq_length'])),
                n_classes=N_CLASSES,
                kernel_size=module_model['kernel_size'],
                dropout_rate=module_model['dropout_rate'])
    #resnet.to(device)  
    #print(resnet)
    tqdm.write("Done!")
    ckpt = torch.load(os.path.join(folder, f'best_model.pth'), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    pred_score = np.zeros((n_total, ))

    for n in tqdm(range(n_batches)):
        start=end
        end = min((n + 1) * setup['batch_size'], n_total)
        with torch.no_grad():
            x = torch.tensor(test_traces[start:end, :,:]).transpose(-1, -2)
            x = x.to(device, dtype=torch.float32)
            score = model(x)
        pred_score[start:end] = score.detach().cpu().numpy().flatten()
        
    # Save predictions
    test_df[f'pred_score'] = pred_score
    
    save_folder = os.path.join("/home/ubuntu/ecg-echo/sjeom/results", module_model['model_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    saved_file = os.path.join(save_folder, f"{module_model['model_name']}_IntTest_predicted-score.csv")
    test_df.to_csv((saved_file), index=False)