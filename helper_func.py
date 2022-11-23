from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm


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

    # data_loader
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_, _, _ in tqdm(train_loader):
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
        