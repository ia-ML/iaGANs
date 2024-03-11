""" 
Simple GAN using fully connected layers

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version

"""
#Note: to watch the progress run:
#    tensorboard --host=<host ip> --logdir=<iaGANs/r00GAN/ptSrc/logs/fake>'

# TODO: 
# Save the images 
# Create a notebook 
# create a generalized demo to control:
#   - dataset
#   - model input and output 
#   - hyper-parameters

import os, time
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


print("torch version: ", torch.__version__)
print("torch GPU    : ", torch.cuda.is_available())
print("TF    version: ", tf.__version__)
print("TF GPU       : ", tf.config.list_physical_devices('GPU'))
#torch version:  2.2.1+cu121
#torch GPU    :  True
#TF    version:  2.16.1
#TF GPU       :  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(), # to get a probablity between 0 and 1
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Is GPU available: ", device)
lr         = 3e-4
z_dim      = 64
image_dim  = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50# 50

disc  = Discriminator(image_dim).to(device)
gen   = Generator(z_dim, image_dim).to(device)

# generate noise 
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# preprocessing
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        #transforms.Normalize((0.1307), (0.3081,)), # mean and sd
    ]
)
# Using MNIST hand written digits dataset
dataset     = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc    = optim.Adam(disc.parameters(), lr=lr)
opt_gen     = optim.Adam(gen.parameters(), lr=lr)
criterion   = nn.BCELoss() # loss function
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
output_folder = "generated_images"  # Name of the output folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start Training 
sTm = time.time()
step = 0
print("Start GAN training ...............")
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real  = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake  = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD      = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                step += 1
 
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                # Convert grid to a matplotlib compatible array 
                img_array = img_grid_fake.permute(1, 2, 0).cpu().numpy() 
                fig, ax = plt.subplots()  # Create a figure and axis
                ax.imshow(img_array)
                ax.axis('off')  # Remove axis for a cleaner look
                plt.savefig(os.path.join(output_folder, f"fake_images_epoch_{epoch}.png"))
                plt.close(fig)  # Close the figure 
                
eTm = time.time()
print("Time: ", eTm-sTm, " seconds!")
print("all tasks are done!")