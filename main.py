# this file is ran separately from telemetry collection

import time
import torch
from models.LeNetCifar10 import preprocess, device, LeNet_Cifar10
from load_data import load_data


data_dir = r".\beamng_dataset-LeNet.pkl"

### PROCESSING AND MODEL PARAMS ###
normalize = False
batch_norm = False
dropout = False
###################################
print(f"Model configuration: device = {device} | normalize = {normalize} | batch_norm = {batch_norm} | dropout = {dropout}")
print('Loading and preprocessing...')
# x_train, y_train, x_test, y_test = load_data(data_dir)
# x_train, x_test = preprocess(x_train, x_test, normalize=normalize)
# x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(data_dir)            # non-speed invocation
x_train, x_valid, x_test = preprocess(x_train, x_valid, x_test, normalize=normalize)          

model = LeNet_Cifar10(batch_norm=batch_norm, dropout=dropout)

start = time.perf_counter()
model.train(x_train, y_train, x_valid, y_valid, 128, 20)
end = time.perf_counter()
print(f"Training duration: {(end - start):.2f} seconds")

accuracy = model.test(x_test, y_test)
with open('test_result.txt', 'w') as f:
    f.write(str(accuracy))
torch.save(model.model.state_dict(), 'model.pt')

print('Test accuracy: %.4f' %accuracy)