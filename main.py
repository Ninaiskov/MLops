import click
import torch
from model import MyAwesomeModel
import torch.nn as nn
import torch.optim as optim
import os

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--dataset", default='corruptmnist', help="dataset to use: corruptmnist or corruptmnist_v2")
@click.option("--trainset_idx", default=0, help="index of the training set to use")
@click.option("--batch_size", default=64, help="batch size to use for training")
@click.option("--shuffle", default=True, help="shuffle the training and test data")

@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")

@click.option("--model_savepath", default='/Users/Nina/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/dtu_mlops/s1_development_environment/exercise_files/final_exercise')


def train(dataset, trainset_idx, batch_size, shuffle, lr, epochs, model_savepath):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainloader, valloader = mnist(dataset=dataset, trainset_idx=trainset_idx, batch_size=batch_size, shuffle=shuffle, split='trainval')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #steps = 0

    trainsize = len(trainloader.dataset)
    valsize = len(valloader.dataset)

    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []

    for e in range(epochs):
        running_loss = 0
        correct = 0
        
        # set model to training mode
        model.train()
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            
            loss = criterion(log_ps, labels)
            running_loss += loss.item() # loss accumulates over batches "running loss"

            loss.backward()
            optimizer.step()
                    
            ps = torch.exp(log_ps)
            preds = torch.argmax(ps, dim=1)
            correct += (preds == labels).sum().item()
        
        else:
            train_loss.append(running_loss/trainsize)    
            train_accuracy.append(correct/trainsize)
            
            ## Validation
            running_loss = 0
            correct = 0
            with torch.no_grad():
                # set model to evaluation mode
                model.eval()
                # validation pass here
                for images, labels in valloader:
                    log_ps = model(images)
            
                    loss = criterion(log_ps, labels)
                    running_loss += loss.item()
                    
                    ps = torch.exp(log_ps)
                    preds = torch.argmax(ps, dim=1)
                    correct += (preds == labels).sum().item() # accumulate correct predictions across batches
                    
        val_loss.append(running_loss/valsize)
        val_accuracy.append(correct/valsize)

        print(f'Epoch {e+1}/{epochs}')    
        print(f'Train accuracy: {train_accuracy[e]*100:.2f}%')  
        print(f'Val accuracy: {val_accuracy[e]*100:.2f}%')
        print('   ')
        print(f'Train loss: {train_loss[e]:.3g}')  
        print(f'Val loss: {val_loss[e]:.3g}')
        print('-----------------')
    
    # save model
    torch.save(model.state_dict(), os.path.join(model_savepath,'trained_model.pt'))
    print('Model saved')

def load_checkpoint(model_savepath, model_checkpoint):
    checkpoint = torch.load(os.path.join(model_savepath,model_checkpoint))
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint)
    return model

@click.command()
@click.argument("model_checkpoint")
@click.option("--model_savepath", default='/Users/Nina/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/dtu_mlops/s1_development_environment/exercise_files/final_exercise')
@click.option("--dataset", default='corruptmnist', help="dataset to use: corruptmnist or corruptmnist_v2")
@click.option("--batch_size", default=64, help="batch size to use for training")
@click.option("--shuffle", default=True, help="shuffle the training and test data")
def evaluate(model_savepath, model_checkpoint, dataset, batch_size, shuffle):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print("Using checkpoint: ", model_checkpoint)

    # TODO: Implement evaluation logic here
    model = load_checkpoint(model_savepath=model_savepath, model_checkpoint=model_checkpoint)
    testloader = mnist(dataset=dataset, trainset_idx=None, batch_size=batch_size, shuffle=shuffle, split='test')
    testsize = len(testloader.dataset)
    test_accuracy = []
    correct = 0
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        # validation pass here
        for images, labels in testloader:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            preds = torch.argmax(ps, dim=1)
            correct += (preds == labels).sum().item() # accumulate correct predictions across batches
            
    test_accuracy = correct/testsize   
    print(f'Test accuracy: {test_accuracy*100:.2f}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
