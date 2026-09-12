# LeNet model for image classification. This model is derived from the standard
# architecture implemented here: []
# The model in this file was modified from the original to serve a driving model
# for BeamNG.drive, which requires multiple outputs rather than a single softmax
# selection. Thse outputs are [steering, throttle, brake]. 
# This model does not solve a classification problem, so Cross Entropy Loss is discarded
# in favor of MSE/SmoothL1 Loss
# TODO: cleanup model.model junk --> merge both classes


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle, tqdm, os
import time

def preprocess(train_images, valid_images, test_images, normalize=False):
    '''
    To preprocess the data by 
        (1).Rescaling the pixels from integers in [0,255) to 
            floats in [0,1), or 
        (2).Normalizing each image using its mean and variance. 

    Args:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.uint8)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.uint8)
        normalize: Boolean. To control to rescale or normalize 
            the images.

    Returns:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.float64)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.float64)
    '''
    ### YOUR CODE HERE
    # cast inputs to floats for type safety
    train_images = train_images.astype(np.float64)
    valid_images = valid_images.astype(np.float64)
    test_images = test_images.astype(np.float64)
    
    # set a small epsilon to avoid div/0
    epsilon = 1e-8
    
    # rescale to floats
    if normalize == False:
        train_images = train_images / 255.0
        valid_images = valid_images / 255.0
        test_images = test_images / 255.0
    
    # normalize with mean and variance
    else:
        # normalize training set
        train_means = np.mean(train_images, axis=(1, 2, 3), keepdims=True) 
        train_var = np.var(train_images, axis=(1, 2, 3), keepdims=True)    
        train_images = (train_images - train_means) / np.sqrt(train_var + epsilon)  
        
        # normalize validation set
        valid_means = np.mean(valid_images, axis=(1, 2, 3), keepdims=True) 
        valid_var = np.var(valid_images, axis=(1, 2, 3), keepdims=True)    
        valid_images = (valid_images - valid_means) / np.sqrt(valid_var + epsilon)    
        
        # normalize test set
        test_means = np.mean(test_images, axis=(1, 2, 3), keepdims=True)
        test_var = np.var(test_images, axis=(1, 2, 3), keepdims=True)
        test_images = (test_images - test_means) / np.sqrt(test_var + epsilon)

    ### END CODE HERE
    return train_images, valid_images, test_images


class LeNet(nn.Module):
    '''
    Build the LeCun network according to the architecture in the homework part 4(c)

    You are free to use the listed APIs from torch.nn:
        torch.nn.Conv2d
        torch.nn.MaxPool2d
        torch.nn.Linear
        torch.nn.ReLU (or other activations)
        torch.nn.BatchNorm2d
        torch.nn.BatchNorm1d
        torch.nn.Dropout

    Refer to https://pytorch.org/docs/stable/nn.html
    for the instructions for those APIs
    '''
    def __init__(self, batch_norm=True, dropout=True):
        # Modifications:
        # (1) removal of n_classes argument, we only care about 3: [steering, throttle, brake]
        # (2) update final layer to have 3 outputs
        
        super(LeNet, self).__init__()
        '''
        Define each layers of the model in __init__() function
        '''

        ### YOUR CODE HERE
        # ========== CLASS PARAMS ==========
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.relu = nn.ReLU()
        
        # ========== MODEL LAYERS ==========
        # 6 out channel convolution (and related operations)
        # note: input is 3x32x32 --> 3 channel input
        # note: output is 6x28x28 --> 6 channel output
        # note: spatial transform 32x32 --> 28x28 requires 5x5 filter (kernel_size) w/ stride=1 and padding=0
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)       
        self.bn1 = nn.BatchNorm2d(num_features=6)
        # relu happens here
        
        # 1st max pool layer (and related operations)
        # note: spatial transform 28x28 --> 14x14 requires 2x2 filter with stride=2
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)   
        
        # 16 out channel convolution (and related operations)
        # note: input is 6x14x14 --> 6 channel input
        # note: output is 16x10x10 --> 16 channel output
        # note: spatial transform 14x14 --> 10x10 requires 5x5 filter (kernel_size) w/ stride=1 and padding=0
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)       
        self.bn3 = nn.BatchNorm2d(num_features=16)
        # relu happens here
        
        # 2nd max pool layer (and related operations)
        # note: spatial transform 10x10 --> 5x5 requires 2x2 filter with stride=2
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)   
        # reshape happens here
        
        # 120 out units; first fully connected layer (and related operations)
        # note: previous maxpool layer has a 5x5x16 output --> 400 element input
        # note: data will have been flattened between pooling and this layer for 1D BN
        self.fc5 = nn.Linear(in_features=400, out_features=120)      
        self.bn5 = nn.BatchNorm1d(num_features=120)
        # relu happens here
        
        # 84 out units; second fully connected layer (and related operations)
        # note: previous fc layer has 128 output --> 128 input
        self.fc6 = nn.Linear(in_features=120, out_features=84)  
        self.bn6 = nn.BatchNorm1d(num_features=84)    
        # relu happens here
        self.drop6 = nn.Dropout()
        
        # 3 out units; output layer (and related operations)
        # note: previous fc layer has 84 output --> 84 input
        self.out = nn.Linear(in_features=84, out_features=3)      
        ### END CODE HERE
    
    def forward(self, x):
        '''
        Run forward pass of the model defined in the above __init__() function
        Args:
            x: Tensor of shape [None, 3, 32, 32]
            for input images.

        Returns:
            logits: Tensor of shape [None, n_classes].
        '''
        
        # Modifications:
        # (1) enforce bounding on the 3 outputs: [steering, throttle, brake]
        # (2) return those bounds
        
                
        ### YOUR CODE HERE
        # (1) Convolution (6 out channels) → BN → ReLU →
        x = self.c1(x)
        if self.batch_norm: x = self.bn1(x)
        x = self.relu(x)
        
        # (2) Max Pooling →
        x = self.mp2(x)
        
        # (3) Convolution (16 out channels) → BN → ReLU →
        x = self.c3(x)
        if self.batch_norm: x = self.bn3(x)
        x = self.relu(x)
        
        # (4) Max Pooling → Reshape to vector →
        x = self.mp4(x)
        x = x.view(x.size(0), -1)
        
        # (5) Fully-connected (120 out units) → BN → ReLU →
        x = self.fc5(x)
        if self.batch_norm: x = self.bn5(x)
        x = self.relu(x)
         
        # (6) Fully-connected (84 out units) → BN → ReLU → Dropout → 
        x = self.fc6(x)
        if self.batch_norm: x = self.bn6(x)
        x = self.relu(x)
        if self.dropout: x = self.drop6(x)
        
        # (7) Outputs (n_classes out units)
        x = self.out(x)
        
        # (8) bound the outputs to real BeamNG inptus
        steering = torch.tanh(x[:, 0:1])        # Bound between [-1, 1]: left/right
        throttle = torch.sigmoid(x[:, 1:2])     # Bound between [0, 1]
        brake = torch.sigmoid(x[:, 2:3])        # Bound between [0, 1]
        
        ### END CODE HERE
        return torch.cat([steering, throttle, brake], dim=1)

# switch between CPU and GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.mps.is_available()
    else "cpu"
)
class LeNet_Cifar10(nn.Module):    
    def __init__(self, criterion=nn.MSELoss(), batch_norm=True, dropout=True):
        # Modifications:
        # (1) remove all n_classes instances
        # (2) switch to MSE/SmoothL1Loss, Cross Entropy is for classification problems, which this is not
        
        if isinstance(criterion, nn.MSELoss) == False and isinstance(criterion, nn.SmoothL1Loss) == False:
            raise ValueError("criterion should be MSE or SmoothL1 Loss types")
        else:
            print(f"Using {"MSE Loss" if isinstance(criterion, nn.MSELoss) else "SmoothL1Loss"} as the criterion")
        
        super(LeNet_Cifar10, self).__init__()
        
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.model = LeNet(batch_norm=batch_norm, dropout=dropout).to(device)  # send to cpu/gpu
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, max_epoch):
        num_samples = x_train.shape[0]
        num_batches = int(num_samples / batch_size)

        num_valid_samples = x_valid.shape[0]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_valid = torch.from_numpy(x_valid).float()
        y_valid = torch.from_numpy(y_valid).float()

        print('---Run...')
        for epoch in range(1, max_epoch + 1):
            self.model.train()
            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()

                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[start:end].to(device)  # send to cpu/gpu
                y_batch = curr_y_train[start:end].to(device)  # send to cpu/gpu

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f}'.format(
                            epoch, loss))

            # To start validation at the end of each epoch.
            self.model.eval()
            valid_loss = 0.0
            print('Doing validation...', end=' ')
            with torch.no_grad():
                for i in range(num_valid_batches):

                    start = batch_size * i
                    end = min(batch_size * (i + 1), x_valid.shape[0])
                    x_valid_batch = x_valid[start:end].to(device)  # send to cpu/gpu
                    y_valid_batch = y_valid[start:end].to(device)  # send to cpu/gpu

                    # Old Cross Entropy code
                    # outputs = self.model(x_valid_batch)
                    # _, predicted = torch.max(outputs.data, 1)
                    # total += y_valid_batch.shape[0]
                    # correct += (predicted == y_valid_batch).sum().item()
                    
                    # New MSE/SmoothL1 code
                    outputs = self.model(x_valid_batch)
                    loss = self.criterion(outputs, y_valid_batch)
                    valid_loss += loss.item() * x_valid_batch.shape[0]

            # Old Cross Entropy code
            # acc = correct / total
            # print('Validation Acc {:.4f}'.format(acc))
            
            # New MSE/SmoothL1 code
            avg_valid_loss = valid_loss / num_valid_samples
            print('Validation Loss {:.6f}'.format(avg_valid_loss))

    def test(self, X_test, y_test):
        self.model.eval()

        # New MSE/SmoothL1 code
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)
        
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test)
            
        # Mean Absolute Error (MAE) for each control
        mae = torch.mean(torch.abs(outputs - y_test), dim=0)
        print('Test MAE (Mean Absolute Error):')
        print('\tSteering:\t{:.6f}'.format(mae[0].item()))
        print('\tThrottle:\t{:.6f}'.format(mae[1].item()))
        print('\tBrake:\t\t{:.6f}'.format(mae[2].item()))
        overall_mae = torch.mean(torch.abs(outputs - y_test))
        print('\tOverall:\t{:.6f}'.format(overall_mae.item()))
        
        return loss.item()

        # Old Cross Entropy code
        # X_test = torch.from_numpy(X_test).float()
        # y_test = torch.from_numpy(y_test)
        # accs = 0
        # for X, y in zip(X_test, y_test):
        #     # modified for running on GPU
        #     X = X.unsqueeze(0).to(device)  # send to cpu/gpu
        #     outputs = self.model(X)        # send to cpu/gpu
            
        #     # original code
        #     # outputs = self.model(X.unsqueeze(0))
            
        #     _, predicted = torch.max(outputs.data, 1)
        #     accs += (predicted == y).sum().item()

        # accuracy = float(accs) / len(y_test)
        
        # return accuracy