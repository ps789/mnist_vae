import torch
import torchvision
import torch.optim as optim
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import model
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')

epochs = 20
batch_size = 64
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_trainset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(
    mnist_trainset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    mnist_testset,
    batch_size=batch_size,
    shuffle=False
)

model = model.LinearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, mu, log_var):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def fit(dataloader):
    model.train()
    for i, data in enumerate(dataloader):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, log_var = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, log_var)
        loss.backward()
        optimizer.step()
    return

for epoch in range(epochs):
    fit(train_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, log_var = model(data)
            if i == int(len(mnist_testset)/test_loader.batch_size) - 1:
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"../outputs/output{epoch}.png", nrow=8)
