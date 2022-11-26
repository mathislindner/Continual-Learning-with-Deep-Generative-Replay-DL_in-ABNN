from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
import random


def test_model(model, test_loader):
    model.eval()
    correct = 0
    for x_, y_ in test_loader:
        x_ = x_.view(-1, 28 * 28)
        x_, y_ = Variable(x_), Variable(y_)
        output = model(x_)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_.data.view_as(pred)).cpu().sum()
    accuracy = correct / len(test_loader.dataset)
    print('Test Accuracy of the model on the test images: {} %'.format( 100 * correct / len(test_loader.dataset)))
    return accuracy

def get_datasets(n, transform):
    random.seed(42)
    # randomly select 2 out of 10 classes
    classes = random.sample(range(10), 10)
    classes_train = classes[2*n:2*n+2] # 2 classes for training
    classes_test = classes[:2*n+2] #all classes seen so far

    MNIST_train_full = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    MNIST_test_full = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    #get indices of the classes we want
    train_indices = [i for i, x in enumerate(MNIST_train_full.targets) if x in classes_train]
    test_indices = [i for i, x in enumerate(MNIST_test_full.targets) if x in classes_test]

    #create new datasets with only the classes we want
    #get subset of the data
    MNIST_train = MNIST_train_full.data[train_indices].float()
    MNIST_train_targets = MNIST_train_full.targets[train_indices]
    MNIST_test = MNIST_test_full.data[test_indices].float()
    MNIST_test_targets = MNIST_test_full.targets[test_indices]

    #create new datasets
    MNIST_train = DS_from_tensors(MNIST_train, MNIST_train_targets)
    MNIST_test = DS_from_tensors(MNIST_test, MNIST_test_targets)
    
    return MNIST_train, MNIST_test

class DS_from_tensors(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
    

def generate_data(G,  classifier, n_images, batch_size = 128):
    #create fake images, label them and return them as a dataset of tensors
    loops = int(n_images/batch_size)
    G.eval()
    classifier.eval()
    fake_images = torch.zeros((loops*batch_size, 28, 28))
    fake_labels = torch.zeros((loops*batch_size))
    for i in range(loops):
        fixed_noise = torch.randn(batch_size, 100)
        new_images = G(fixed_noise)
        pred = classifier(new_images)
        new_labels = pred.data.max(1, keepdim=True)[1]
        fake_images[i*batch_size:(i+1)*batch_size] = new_images.reshape(batch_size, 28, 28)
        fake_labels[i*batch_size:(i+1)*batch_size] = new_labels.reshape(batch_size)

    dataset = DS_from_tensors(fake_images, fake_labels.long())
    return dataset


def get_accuracies(evaluations):
    accuracies = []
    for i, evaluation in enumerate(evaluations):
        temp_acurracies = []
        j = -1
        while j < i:
            j += 1
            key_temp = 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{}'.format(j)
            temp_acurracies.append(evaluation[key_temp])
        accuracies.append(sum(temp_acurracies)/len(temp_acurracies))
    return accuracies

# create a grid of 5x5 images
def generate_images(G,  n_images = 25):
    G.eval()
    fixed_noise = torch.randn(n_images, 100)
    test_images = G(fixed_noise)
    test_images = test_images.view(test_images.size(0), 1, 28, 28)
    test_images = test_images.data
    grid = torchvision.utils.make_grid(test_images)
    return grid

def train_GAN(G, D, train_loader, G_optimizer, D_optimizer, BCE_loss,  train_epoch):
    G.train()
    D.train()

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_, _ in train_loader:
            #wrap in variable
            x_ = Variable(x_)

            # add minor noise to the data
            x_ = x_ + 0.5 * torch.randn(x_.size())

            D.zero_grad()
            # convert x_ to tensor
            #x_ = torch.tensor(x_).view(-1, 28 * 28)
            x_ = x_.view(-1, 28 * 28)
            mini_batch = x_.size()[0]
            y_real_ = torch.ones(mini_batch,1)
            y_fake_ = torch.zeros(mini_batch,1)

            x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)

            D_result = D(x_)
            D_real_loss = BCE_loss(D_result, y_real_)
            D_real_score = D_result

            z_ = torch.randn(mini_batch, 100)
            z_ = Variable(z_)
            G_result = G(z_)

            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            # train generator G
            G.zero_grad()

            z_ = torch.randn(mini_batch, 100)
            y_ = torch.ones(mini_batch, 1)

            z_, y_ = Variable(z_), Variable(y_)
            G_result = G(z_)
            D_result = D(G_result)
            G_train_loss = BCE_loss(D_result, y_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)
        print('[%d/%d]:GAN loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))    
    return G, D, D_losses, G_losses
        