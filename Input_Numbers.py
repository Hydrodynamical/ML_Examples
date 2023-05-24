#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame # for user drawn input
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.functional as F


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(torch.cuda.get_device_name(0) if device.type == "cuda" else "Apple Neural Engine")

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.75)
        self.dropout2 = nn.Dropout(0.75)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        #print(x.size())
        x = torch.flatten(x, 1)
        #print(x.size())
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


# In[3]:


model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("MNIST_CNN.pth"))


# In[4]:


def print_array(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] > 0.5:
                print('&', end='')
            else:
                print(' ', end='')
        print()

def flip_array(arr):
    copy_arr = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i,j] = copy_arr[j, i]
    return arr
    
def downscale(arr, R):
    downscaled_arr = np.zeros([28,28])
    for i in range(28):
        for j in range(28):
            for k in range(R):
                for l in range(R):
                    downscaled_arr[i,j] += (arr[i*R + k, j*R + l]/(R ** 2))
    return downscaled_arr

def normalize(arr):
    # Normalize so that 0 is white and 1 is black
    normalized_arr = np.zeros([28,28])
    for i in range(28):
        for j in range(28):
            normalized_arr[i,j] = int(255- arr[i,j])/255
    
    return normalized_arr

def prepared_array(arr):
    # Convert the array to a numpy array
    np_arr_bw = np.asarray(arr)

    # Mollify bw array to 28 by 28
    downscaled_arr = downscale(np_arr_bw, R)

    # Normalize so that 0 is white and 255 is black
    normalized_arr = normalize(downscaled_arr)

    # Orient array correctly
    flipped_arr = flip_array(normalized_arr)
    
    final_arr = np.zeros([1, 28, 28])
    final_arr[0,:,:] = flipped_arr[:,:]

    return final_arr

# Initialize Pygame
pygame.init()

# Set the screen size
R = 2
screen = pygame.display.set_mode((28*R, 28*R))

# Set the background color to white
background = (255, 255, 255)
screen.fill(background)

# Set the initial position of the mouse
prev_pos = (0, 0)

# Set the drawing color to black
color = (0, 0, 0)

# Set the drawing thickness
thickness = 4

# Create a clock object to control the frame rate
clock = pygame.time.Clock()

# Set the font and font size
font = pygame.font.Font(None, 12)

# Game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit the game
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                screen.fill(background)
            elif event.key == pygame.K_RETURN:
                # Get the current display as an array
                arr = pygame.surfarray.pixels_green(screen)
                
                # Prep the array for the network
                final_arr = prepared_array(arr)
                
                # Print picture
                figure = plt.figure(figsize = (3,3))
                plt.axis("off")
                plt.imshow(final_arr.squeeze(), cmap = "gray")
                plt.show()
                
                # User input as a Torch Tensor
                x = torch.from_numpy(final_arr)
                x = x.float()
                x = torch.unsqueeze(x, dim = 0)
                
                # Calculate network on user input
                model.eval()
                with torch.no_grad():
                    x = x.to(device)
                    pred = model(x)
                    predicted = pred[0].argmax(0)
                    print(pred[0])
                    print(f"Predicted: {predicted}.")
                
    # Get the current position of the mouse
    pos = pygame.mouse.get_pos()

    # Check if the left mouse button is pressed
    if pygame.mouse.get_pressed()[0]:
        # Draw a line from the previous position to the current position
        pygame.draw.line(screen, color, prev_pos, pos, thickness)

    # Update the screen
    pygame.display.update()

    # Set the previous position of the mouse
    prev_pos = pos
    
    # Limit the frame rate to 60 fps
    clock.tick(180)

