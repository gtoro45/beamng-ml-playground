# LeNet model for image classification. This model is derived from the standard
# architecture implemented here: []
# The model in this file was modified from the original to serve a driving model
# for BeamNG.drive, which requires multiple outputs rather than a single softmax
# selection. Thse outputs are [steering, throttle, brake]. 
# This model does not solve a classification problem, so Cross Entropy Loss is discarded
# in favor of MSE/SmoothL1 Loss
#
# There are 2 models in this file: an image variant, and an image + speed variant.
# Both of these models inherit from their respective LeNetCifar10 counterparts, and are
# modified for stacked frame input for temporal awareness

from torch import nn
from models.LeNetCifar10 import LeNet as ImageLenNet
from models.LeNetCifar10_Speed import LeNet as SpeedLeNet

class StackedLeNet(ImageLenNet):
    def __init__(self, stack_size, **kwargs):
        super().__init__(**kwargs)

        # We only modify the input layer, all else stays the same
        # Add 3 * K (stack_size) so input shape is [N, 3*K, H, W]
        
        # 6 out channel convolution (and related operations)
        # note: input is 3*Kx32x32 --> 3*K channel input
        # note: output is 6x28x28 --> 6 channel output
        # note: spatial transform 32x32 --> 28x28 requires 5x5 filter (kernel_size) w/ stride=1 and padding=0
        self.c1 = nn.Conv2d(in_channels=3 * stack_size, out_channels=6, kernel_size=5, stride=1)    

class StackedSpeedLeNet(SpeedLeNet):
   def __init__(self, stack_size, **kwargs):
        super().__init__(**kwargs)

        # We only modify the input layer, all else stays the same
        # Add 3 * K (stack_size) so input shape is [N, 3*K, H, W]
        
        # 6 out channel convolution (and related operations)
        # note: input is 3*Kx32x32 --> 3*K channel input
        # note: output is 6x28x28 --> 6 channel output
        # note: spatial transform 32x32 --> 28x28 requires 5x5 filter (kernel_size) w/ stride=1 and padding=0
        self.c1 = nn.Conv2d(in_channels=3 * stack_size, out_channels=6, kernel_size=5, stride=1)    
