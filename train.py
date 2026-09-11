# this file is ran separately from telemetry collection

import time
import torch
import torch.nn as nn
import argparse
from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Select the model to train")
parser.add_argument('--dataset', type=str, required=True, help="The dataset to train on")
args = parser.parse_args()
MODEL = args.model
DATA = args.dataset

def train_lenet_cifar_10(dataset_file_path):
    from models.LeNetCifar10 import preprocess, device, LeNet_Cifar10

    ### PROCESSING AND MODEL PARAMS ###
    normalize = True
    batch_norm = True
    dropout = True
    ###################################
    print(f"Model configuration: device = {device} | normalize = {normalize} | batch_norm = {batch_norm} | dropout = {dropout}")
    print('Loading and preprocessing...')
    # x_train, y_train, x_test, y_test = load_data(dataset_file_path)
    # x_train, x_test = preprocess(x_train, x_test, normalize=normalize)
    # x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(dataset_file_path)            # non-speed invocation
    x_train, x_valid, x_test = preprocess(x_train, x_valid, x_test, normalize=normalize)          

    # switch between MSE and SmoothL1
    model = LeNet_Cifar10(criterion=nn.SmoothL1Loss(), batch_norm=batch_norm, dropout=dropout)

    start = time.perf_counter()
    model.train(x_train, y_train, x_valid, y_valid, 128, 20)
    end = time.perf_counter()
    print(f"Training duration: {(end - start):.2f} seconds")

    test_loss = model.test(x_test, y_test)
    torch.save(model.model.state_dict(), 'lenet.pt')

    print('Test loss: %.6f' %test_loss)
    
def train_lenet_cifar_10_speed(dataset_file_path):
    from models.LeNetCifar10_Speed import preprocess, device, LeNet_Cifar10_Speed

    ### PROCESSING AND MODEL PARAMS ###
    normalize = True
    batch_norm = True
    dropout = True
    ###################################
    print(f"Model configuration: device = {device} | normalize = {normalize} | batch_norm = {batch_norm} | dropout = {dropout}")
    print('Loading and preprocessing...')
    # x_train, y_train, x_test, y_test = load_data(dataset_file_path)
    # x_train, x_test = preprocess(x_train, x_test, normalize=normalize)
    # x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(dataset_file_path, use_speed_as_input=True)            # non-speed invocation
    
    x_train_img, x_train_speed = x_train
    x_valid_img, x_valid_speed = x_valid
    x_test_img, x_test_speed = x_test
    
    # preprocess images only
    x_train_img, x_valid_img, x_test_img = preprocess(x_train_img, x_valid_img, x_test_img, normalize=normalize)      

    # repackage images with speed
    x_train = (x_train_img, x_train_speed)
    x_valid = (x_valid_img, x_valid_speed)
    x_test = (x_test_img, x_test_speed)
    
    # switch between MSE and SmoothL1
    model = LeNet_Cifar10_Speed(criterion=nn.SmoothL1Loss(), batch_norm=batch_norm, dropout=dropout)

    start = time.perf_counter()
    model.train(x_train, y_train, x_valid, y_valid, 128, 20)
    end = time.perf_counter()
    print(f"Training duration: {(end - start):.2f} seconds")

    test_loss = model.test(x_test, y_test)
    torch.save(model.model.state_dict(), 'lenet-speed.pt')

    print('Test loss: %.6f' %test_loss)
    
if __name__ == "__main__":
    match MODEL:
        case "lenet":       train_lenet_cifar_10(DATA)
        case "lenet-speed": train_lenet_cifar_10_speed(DATA)