from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms.v2 as transforms
import pygame
import numpy as np
from scipy.ndimage import gaussian_filter

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_features=15488, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        #Block 1
        x = self.conv1(x)
        x = F.relu(x)

        #Block 2
        x = self.conv2(x)
        x = F.relu(x)

        #Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        #Dropout and Flatten
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        #Block 4
        x = self.fc1(x)
        x = F.relu(x)

        #Block 5 SOFTMAX
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #reset optimizer
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx * len(data)) % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #Sum loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            #Find prediction with highest value
            pred = output.argmax(dim=1, keepdim=True)
            #Check if it is equal to predicted value
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss
    
def train_cnn(num_epochs):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    save_dir = 'saved_weights/' + timestamp
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {save_dir}")
        return

    BATCH_SIZE = 128
    NUM_WORKERS = 4

    preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=30, translate=(0.2,0.2), scale=(0.9, 1.1), shear=10)
        ])

    dataset_directory = ".." + os.sep + "data" + os.sep

    trainingset = torchvision.datasets.MNIST(root=dataset_directory, train=True, download=True, transform=preprocessing)
    training_loader = torch.utils.data.DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    testingset = torchvision.datasets.MNIST(root=dataset_directory, train=False, download=True, transform=preprocessing)
    testing_loader = torch.utils.data.DataLoader(dataset=testingset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device=device)
    optimizer = optim.Adam(params=model.parameters())

    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.1)

    args = None

    best_loss = float('inf')
    print("\nStarting Training")
    for epoch in range(num_epochs):
        train(args, model, device, training_loader, optimizer, epoch)
        current_loss = test(model, device, testing_loader)
        scheduler.step()
    
        if current_loss < best_loss:
            best_loss = current_loss

            now = datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            destination = os.path.join(save_dir, str(epoch) + "_mnist_cnn.pt")
            try: 
                torch.save(model.state_dict(), destination)
            except Exception as e:
                print("COULD NOT SAVE WEIGHTS\n")
                return


def parse_args():
    #Create Parser
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #train = Train model
    #draw = draw number to use model 
    parser.add_argument(
        '--task',
        required=True,
        choices=['train', 'draw'],
        help="Which task would you like to preform, training or drawing"
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        help="Required when task is training. Number of epochs"
    )
    parser.add_argument(
        '--load_weights',
        help="Required when task is drawing. Path to the weights file to load.\nShould follow saved_weights/TIMESTAMP/FILENAME.pt ."
    )
    
    args = parser.parse_args()

    if args.task == 'draw' and args.load_weights is None:
        parser.error("--load_weights is required when --task is draw")

    if args.task == 'draw' and args.num_epochs is not None:
        parser.error("--num_epochs is required when --task is train, not when task is draw")

    if args.task == 'train' and args.load_weights is not None:
        parser.error("--load_weights is required when --task is draw, not when task is train.")

    if args.task == 'train' and args.num_epochs is None:
        parser.error('--num_epochs is required when --task is train.')

    return args

def main():
    ARGS = parse_args()
    if ARGS.task == 'train':
        train_cnn(ARGS.num_epochs)
    else:
        model = Model()
        try:
            model.load_state_dict(torch.load(ARGS.load_weights))
        except Exception as e:
            print(f"Failed to load weights from {ARGS.load_weights}. Error {e}")
            return 
        
        drawer(model)

        return


def drawer(model):
    pygame.init()

    pixel_height, pixel_width = 28, 28
    scale = 20

    screen = pygame.display.set_mode((pixel_height*scale, pixel_width*scale))

    pixels = np.zeros((pixel_height, pixel_width))

    running = True
    dragging = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Grayscale(),
            transforms.Resize((28, 28)), #should be 28, 28 anyway
            transforms.ToTensor()
        ])

        processed_image = preprocess(pixels.astype(np.float32))
        processed_image = processed_image.unsqueeze(0)
        model.eval()

        with torch.no_grad():
            output = model(processed_image)

        predicted_class = output.argmax().item()

        print(predicted_class)

        if dragging:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < pixel_width*scale and 0 <= y < pixel_height*scale:  # Check if mouse position is within window
                row, col = y // scale, x // scale
                if row >= 0 and row < pixel_height and col >= 0 and col < pixel_width:
                    pixels[row][col] = min(255, pixels[row][col] + 50)

                if row >= 0 and row < pixel_height and col - 1 >= 0 and col < pixel_width: #left
                    pixels[row][col - 1] = min(255, pixels[row][col - 1] + 5)

                if row >= 0 and row < pixel_height and col + 1 < pixel_width: # right
                    pixels[row][col + 1] = min(255, pixels[row][col + 1] + 5)

                if row - 1 >= 0 and col >= 0 and col < pixel_width: #up
                    pixels[row - 1][col] = min(255, pixels[row - 1][col] + 5)

                if row + 1 < pixel_height and col >= 0 and col < pixel_width: #down
                    pixels[row + 1][col] = min(255, pixels[row + 1][col] + 5)
                
                #7x7 area around clicked pixel
                top = max(0, row - 3)
                bottom = min(pixel_height, row + 4)
                left = max(0, col - 3)
                right = min(pixel_width, col + 4)
                region = pixels[top:bottom, left:right]
                blurred_region = gaussian_filter(region, sigma=0.24)
                pixels[top:bottom, left:right] = blurred_region

        #Draw screen
        for row in range(pixel_height):
            for col in range(pixel_width):
                color = (pixels[row][col], pixels[row][col], pixels[row][col])
                pygame.draw.rect(screen, color, pygame.Rect(col*scale, row*scale, scale, scale))

        pygame.display.flip()

if __name__ == '__main__':
    main()