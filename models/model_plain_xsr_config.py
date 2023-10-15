# Image Conf
R = 224
IMAGE_SIZE = (R, R)
L = 96
SUB_IMAGE_SIZE = (L, L)
# Network Conf
NUM_EPOCHS = 2000
LR = 1e-4
NUM_CHANNELS = 6
KERNEL = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)
# Checkpoint
SAVE_ITER_FILE = f'weights/weights_net_{L}_iter'
SAVE_EPOCH_FILE = f'weights/weights_net_{L}_epoch'
SAVE_FINISH_FILE = f'weights/weights_final_net_{L}_epoch'
