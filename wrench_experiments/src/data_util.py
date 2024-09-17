import numpy as np
import os
import torch

DEFAULT_DATA_PATH = '/mnt/e/datasets'

def load_wrench_dataset(dataset_name, data_base_path=DEFAULT_DATA_PATH, use_torch=False):
    """
    load wrench dataset
    The only difference from load_dataset is that a_train, a_test don't exist
    """

    data_folder_name = get_data_folder_name(dataset_name)
    
    # load dataset
    train = np.load(os.path.join(data_base_path, data_folder_name, 
                                    'train.npz'),
                    allow_pickle=True)
    test = np.load(os.path.join(data_base_path, data_folder_name,
                                'test.npz'),
                    allow_pickle=True)

    # unpack dataset
    x_train, y_train = train['x'], train['y']
    x_test, y_test = test['x'], test['y']
    
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    
    # np to torch
    if use_torch:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
        
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()
    return x_train, y_train, x_test, y_test


def load_LF(dataset_name, data_base_path=DEFAULT_DATA_PATH):
    data_folder_name = get_data_folder_name(dataset_name)
    L = np.load(os.path.join(data_base_path, data_folder_name, 'LF.npy'))
    
    return L


def get_data_folder_name(dataset_name):
    return dataset_name.lower()
 