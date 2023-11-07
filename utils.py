# Generate a video

import torch
import imageio
import numpy as np
from typing import Iterable

# Suppose 'tensors' is your list of image tensors
# Ensure they are in the correct format (uint8) with values in [0.0, 1.0]

def output_video(image_tensors: Iterable[torch.Tensor], output_file='output_video.mp4', fps=25):
    # Create a video writer object using imageio
    with imageio.get_writer(output_file, fps=fps) as video:
        for tensor in image_tensors:
            # Convert PyTorch tensor to numpy array
            # If your tensor is between 0 and 1, you should multiply by 255 and cast to uint8
            # If your tensor is already in the 0-255 range, just cast to uint8
            numpy_image = (tensor.squeeze(0).cpu().detach().numpy() * 255).astype(np.uint8)
            
            # If tensor is in CHW format (channels, height, width), convert it to HWC
            if numpy_image.shape[0] in {1, 3}:
                # Handle grayscale images (1 channel) or RGB images (3 channels)
                numpy_image = np.transpose(numpy_image, (1, 2, 0))
            
            video.append_data(numpy_image)