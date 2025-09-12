import gyrointerp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch


def gyro_data(ages, teff_err=75, prot_err=0.1, nsize=100, plot=False):
    all_data = []  # to hold arrays of shape (nsize, 2) for each age

    for age in ages:
        # Generate teff and noisy teff
        teffs = np.linspace(3800, 6200, nsize)
        teffs_n = teffs + np.random.normal(0, teff_err, nsize)
        teffs_n = np.clip(teffs_n, 3800, 6200)

        # Generate rotation periods from gyro model
        prots = gyrointerp.models.slow_sequence(teffs, age)
        prots_n = gyrointerp.models.slow_sequence(teffs_n, age) + np.random.normal(
            0, gyrointerp.models.slow_sequence(teffs_n, age) * prot_err, nsize
        )

        # Optionally show scatter plot
        if plot:
            plt.scatter(teffs_n, prots_n)
            plt.gca().invert_xaxis()
            plt.title(f'Age = {age}')
            plt.xlabel('Teff (K)')
            plt.ylabel('Prot (days)')
            plt.show()

        # Stack teff and prot into shape (nsize, 2)
        data = np.column_stack((teffs_n, prots_n))
        all_data.append(data)

    return np.array(all_data), np.array(ages)

def split_dataset(all_data, all_labels, size=0.2, seed=42, batch_size=64):
    
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(all_data, all_labels, test_size=size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.5, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.double)
    y_train_tensor = torch.tensor(y_train, dtype=torch.double) 

    X_val_tensor = torch.tensor(X_val, dtype=torch.double)
    y_val_tensor = torch.tensor(y_val, dtype=torch.double)

    X_test_tensor = torch.tensor(X_test, dtype=torch.double)
    y_test_tensor = torch.tensor(y_test, dtype=torch.double)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    # Create PyTorch DataLoader instances
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader