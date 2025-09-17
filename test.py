import torch
import torch.optim as optim
import net 

def main():
    # loading in a checkpoint...

    # loads in the checkpoint from a pth we saved earlier 
    checkpoint = torch.load("checkpoint.pth")
    # creates a blank model for the checkpoint to be loadedinto 
    model = net.StellarClusterCNN()
    # loads the checkpoint into the blank model
    model.load_state_dict(checkpoint["model_state_dict"])

    # make sure this is the same as in train() !!! 
    # TODO: this hardcoded (BAD!) and should be fixed
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # TODO: load in the real data and normalize it properly
    # TODO: pass the real data through the model and analyze 
    
    
    # start_epoch = checkpoint["epoch"]
    # train_loss_at_save = checkpoint["loss"]
    
if __name__ == "__main__":
    main()