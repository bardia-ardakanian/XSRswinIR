import torch
import torch.nn as nn
from torchvision import models, transforms
from model_plain_xsr_config import *


def get_score_module(pretrained_weights=f'{SAVE_EPOCH_FILE}_499.pth'):
    """
    Builds and returns a pre-trained texture classifier (based on VGG16 architecture).

    Parameters:
        pretrained_weights (str): Path to the pre-trained model weights.

    Returns:
        model (torch.nn.Module): Pre-trained and evaluated texture classifier model.
    """

    # Build the texture classifier (based on VGG16 architecture)
    model = build_tc()

    # Load pre-trained weights into the VGG16 feature extractor
    vgg16 = model.features.eval()

    # Define the loss function (Mean Squared Error) and the optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Load checkpoint which contains model & optimizer states, epoch info, and iteration info
    model, optimizer, _, _ = load_checkpoint(model, optimizer, f'weights\{pretrained_weights}')

    # Set the model to evaluation mode (deactivates dropout layers, etc.)
    model.eval()

    return model


def build_tc(num_channels=NUM_CHANNELS, sub_image_size=L, kernel=KERNEL, stride=STRIDE, padding=PADDING):
    """
    Builds a texture classifier based on the VGG16 architecture with custom input and output layers.

    Parameters:
        num_channels (int): Number of input channels.
        sub_image_size (int): Size of the sub-image.
        kernel (int): Size of the convolutional kernel.
        stride (int): Stride of the convolutional kernel.
        padding (int): Padding added to the input.

    Returns:
        model (torch.nn.Module): Texture classifier with modified input and output layers.
    """

    # Load a VGG16 model without pre-trained weights
    model = models.vgg16(pretrained=False)

    # Modify the input channels and first convolutional layer to match our requirements
    model.features[0] = nn.Conv2d(num_channels, sub_image_size, kernel_size=kernel, stride=stride, padding=padding)
    model.features[1] = nn.Conv2d(sub_image_size, 64, kernel_size=kernel, stride=stride, padding=padding)

    # Modify the fully-connected layers of the VGG16 to achieve a final output between 0 and 1 (using a sigmoid
    # activation)
    model.classifier[-1] = nn.Sequential(
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),  # Add a Batch Normalization layer
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),  # Add a Batch Normalization layer
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),  # Add a Batch Normalization layer
        nn.Linear(128, 1),
        nn.Sigmoid(),  # Sigmoid activation for binary classification
    )

    # Check for GPU availability and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


def load_checkpoint(model, optimizer, filename):
    """
    Load a model and optimizes states from a saved checkpoint.

    Parameters:
    - model (torch.nn.Module): The model whose states we want to update.
    - optimizer (torch.optim.Optimizer): The optimizer whose states we want to update.
    - filename (str): Path to the checkpoint file (typically with a .pth extension).

    Returns:
    - model (torch.nn.Module): The model updated with states from the checkpoint.
    - optimizer (torch.optim.Optimizer): The optimizer updated with states from the checkpoint.
    - epoch (int): The epoch number at which the training was saved.
    - iteration (int): The iteration number at which the training was saved.
    """
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']

    return model, optimizer, epoch, iteration
