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

def test_model(model, n, test_loader):
    model.eval()
    correct = 0
    for x_, y_ in test_loader:
        x_ = x_.view(-1, 28 * 28)
        x_, y_ = Variable(x_), Variable(y_)
        output = model(x_)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_.data.view_as(pred)).cpu().sum()
    accuracy = correct / len(test_loader.dataset)
    print('Test Accuracy of the model on the {} test images: {} %'.format(n, 100 * correct / len(test_loader.dataset)))
    return accuracy

def get_data_loaders(n, batch_size, transform):
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
    MNIST_train = torch.utils.data.Subset(MNIST_train_full, train_indices)
    MNIST_test = torch.utils.data.Subset(MNIST_test_full, test_indices)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=MNIST_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def generate_data(G, D, n_images, batch_size = 128):
    #create fake images, label them and return them as a dataset of tensors
    loops = int(n_images/batch_size)
    G.eval()
    D.eval()
    fake_images = []
    fake_labels = []
    for i in range(loops):
        z_ = torch.randn((batch_size, 100))
        fake_images.append(G(z_).reshape(batch_size, 28*28))
        fake_labels.append(D(fake_images[i]))
    fake_images = torch.reshape(torch.cat(fake_images), (-1, 1, 28, 28))
    fake_labels = torch.Tensor(torch.cat(fake_labels)).reshape(-1, 1)

    #create dataset
    fake_dataset = torch.utils.data.TensorDataset(fake_images, fake_labels)
    return fake_dataset


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
def generate_images(G, n_images):
    G.eval()
    fixed_z_ = torch.randn((5 * 5, 100))    # fixed noise
    test_images = G(fixed_z_)
    test_images = test_images.view(test_images.size(0), 1, 28, 28)
    test_images = test_images.data
    grid = torchvision.utils.make_grid(test_images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()
    return None

def train_GAN(G, D, train_loader, G_optimizer, D_optimizer, train_epoch):
    G.train()
    D.train()
    fixed_z_ = torch.randn((5 * 5, 100))    # fixed noise
    fixed_z_ = Variable(fixed_z_, volatile=True)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_, _ in train_loader:
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
        