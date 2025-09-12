import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# imports for custom files
import net 
import get_data

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed) 

def normalize(x, var_type = None):
    x2=np.log10(x)
    # x_std=x2
    x_min = np.min(x2)
    x_max = np.max(x2)
    x_norm = (x2 - x_min) / (x_max - x_min)

    with open('norm.txt', 'a') as file:
        file.write(f"\n{var_type} \nmin: {x_min:.8f} \nmax: {x_max:.8f}\n")

    return x_norm

def save_state(epoch, model, opt, loss):
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "loss": loss.item(),
    }, "checkpoint.pth")


def train(model, num_epochs, train_loader, val_loader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.train()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    model.to(device)
    criterion.to(device)

    # Print the model architecture
    # print(model)
    
    # training loop:
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float() 
        
            # important! must always zero out the gradients before the next forward pass
            optimizer.zero_grad()

            # forward propagation
            outputs = model(inputs).squeeze() 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch%100==0:
            print("epoch:", epoch)
            eval(model, val_loader)
            print("Loss: ", loss.item())
            save_state(epoch, model, optimizer, loss)
        
        model.train()

def eval(model, val_loader):
    device = torch.device("cpu")
    model.eval()

    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    print(f'Validation MSE: {avg_loss:.6f}')


def main():
    num_epochs = 10000

    print("loading in data")

    # Load in data
    # raw_X, raw_y = get_data.load_files()
    raw_X, raw_y  = get_data.gyro_data(np.random.uniform(100,1000,200))

    # clears the normalization text file 
    open("norm.txt", "w").close()
    # Normalize data
    normalized_X = np.swapaxes(np.array((normalize(raw_X[:,0], var_type='p_rot'), normalize(raw_X[:,1], var_type='color'))), 0, 1) 
    normalized_y = np.array(normalize(raw_y.squeeze(), var_type='age')) 

    # Split data. Make sure your data is in this format... X shape: (# samples, # dims)  y shape: (# samples ,)
    train_loader, test_loader, val_loader = get_data.split_dataset(normalized_X, normalized_y)

    print("creating a new model")
    # Create an instance of the model
    model = net.StellarClusterCNN()


    print("starting training")
    train(model, num_epochs, train_loader, val_loader)

    print("training complete. \nstarting post-training evaluation.")
    eval(model, test_loader)

    # #loading in a checkpoint
    # checkpoint = torch.load("checkpoint.pth")
    # model = net.StellarClusterCNN()
    # model.load_state_dict(checkpoint["model_state_dict"])

    # optimizer = optim.Adam(model.parameters(), lr=1e-6)   # TODO: change lr
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"]
    # loss_at_save = checkpoint["loss"]

    # model.train()  # Switch back to training mode 

    # train(model, num_epochs, train_loader, val_loader)

    
if __name__ == "__main__":
    main()