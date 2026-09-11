# this file is ran separately from telemetry collection

import time
import torch
import torch.nn as nn
from load_data import load_data


def train_lenet_cifar_10():
    from models.LeNetCifar10 import preprocess, device, LeNet_Cifar10
    data_dir = r".\beamng_dataset-LeNet.pkl"

    ### PROCESSING AND MODEL PARAMS ###
    normalize = True
    batch_norm = True
    dropout = True
    ###################################
    print(f"Model configuration: device = {device} | normalize = {normalize} | batch_norm = {batch_norm} | dropout = {dropout}")
    print('Loading and preprocessing...')
    # x_train, y_train, x_test, y_test = load_data(data_dir)
    # x_train, x_test = preprocess(x_train, x_test, normalize=normalize)
    # x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(data_dir)            # non-speed invocation
    x_train, x_valid, x_test = preprocess(x_train, x_valid, x_test, normalize=normalize)          

    # switch between MSE and SmoothL1
    model = LeNet_Cifar10(criterion=nn.SmoothL1Loss(), batch_norm=batch_norm, dropout=dropout)

    start = time.perf_counter()
    model.train(x_train, y_train, x_valid, y_valid, 128, 20)
    end = time.perf_counter()
    print(f"Training duration: {(end - start):.2f} seconds")

    test_loss = model.test(x_test, y_test)
    torch.save(model.model.state_dict(), 'model.pt')

    print('Test loss: %.6f' %test_loss)