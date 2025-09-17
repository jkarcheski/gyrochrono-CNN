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


def train(model, num_epochs, train_loader, val_loader, lr=1e-6):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.train()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    criterion.to(device)

    # Print the model architecture
    # print(model)

    train_losses = []
    val_losses = []
    
    # training loop:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float() 
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze() 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # sum weighted by batch size

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        val_loss = validation_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if epoch%100==0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}. Validation Loss: {val_loss:.4f}")
            # eval(model, val_loader)
            save_state(epoch, model, optimizer, loss)
        
        model.train()

    return train_losses, val_losses

def validation_loss(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(val_loader.dataset)
    return avg_loss

def plot_losses(train, val, epochs):
    fig, ax = plt.subplots()

    ax.set_title("Loss over time")
    ax.set_xlabel("Epoch")

    ax.plot(range(epochs), train, label='training  loss')
    ax.plot(range(epochs), val, label='validation loss')

    ax.legend()
    
    plt.show()

def main():
    num_epochs = 5000

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
    train_losses, validation_losses = train(model, num_epochs, train_loader, val_loader, lr=1e-5)
    plot_losses(train_losses, validation_losses, num_epochs)

    print("training complete. \nstarting post-training evaluation.")
    eval(model, test_loader)
    
if __name__ == "__main__":
    main()