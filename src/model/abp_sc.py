import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, kernel_size, channel_size, image_size):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.image_size = image_size
        self.initial_size = int(self.image_size / 4)

        self.latent_to_feature = nn.Sequential( # MNIST Input Z dimension
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(), # Output hidden_size

            nn.Linear(self.hidden_size, self.initial_size * self.initial_size * self.kernel_size),
            nn.ReLU() # Output 7x7xkernel_size
        )

        self.feature_to_output = nn.Sequential( # MNIST Input: 7x7xkernel_size
            nn.ConvTranspose2d(in_channels=self.kernel_size, out_channels=self.kernel_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.kernel_size),
            nn.ReLU(), # Output 14x14xkernel_size

            nn.ConvTranspose2d(in_channels=self.kernel_size, out_channels=self.channel_size,kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid() # Output 28x28x1
        )

    def forward(self, z):
        features = self.latent_to_feature(z)
        features = features.view(z.shape[0], self.kernel_size, self.initial_size, self.initial_size)
        output = self.feature_to_output(features)
        output = output.view(z.shape[0], self.channel_size, self.image_size, self.image_size)
        return output


class Linear_Generator(nn.Module):
    def __init__(self, latent_size, hidden_size,  channel_size, image_size):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.channel_size = channel_size
        self.image_size = image_size

        self.latent_to_feature = nn.Sequential( # MNIST Input Z dimension
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(), # Output hidden_size

            nn.Linear(self.hidden_size, self.channel_size * self.image_size * self.image_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        features = self.latent_to_feature(z)
        output = features.view(z.shape[0], self.channel_size, self.image_size, self.image_size)
        return output


class ABPSC():
    def __init__(self, latent_size, alpha, learning_rate, langevin_steps, noise_variance, slab_variance, langevin_stepsize, device):
        super().__init__()

        self.device = device
        self.noise_variance = noise_variance
        self.slab_variance = slab_variance
        self.langevin_steps = langevin_steps
        self.latent_size = latent_size
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.langevin_stepsize = langevin_stepsize

    def init_weight(self):
        classname = self.generator.__class__.__name__
        if classname.find("ConvTranspose2d") != -1:
            nn.init.normal_(self.generator.weight.data, 0.0, 0.001)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(self.generator.weight.data, 1.0, 0.001)
            nn.init.normal_(self.generator.bias.data, 0.0)

    def loss_function(self, data, generated):
        loss = 0.5 / (self.noise_variance ** 2) * torch.pow(data - generated, 2).sum()
        return loss

    def latent_gradient(self, z):
        #exp1 = torch.exp(-torch.pow(z, 2) / 2)
        #exp2 = torch.exp(-torch.pow(z, 2) / 2 / (self.slab_variance ** 2))
        #gradient = -(self.alpha * z * exp1 + (1 - self.alpha) * z / (self.slab_variance ** 3) * exp2) / (self.alpha * exp1 + (1 - self.alpha) / self.slab_variance * exp2)
        ratio1 = (1 - self.alpha) / (self.alpha * self.slab_variance)
        ratio2 = (self.slab_variance * self.slab_variance - 1) / (self.slab_variance * self.slab_variance)
        gradient = - z + ratio1 * ratio2 * z / (torch.exp(-ratio2 * torch.pow(z, 2) / 2) + ratio1)
        return gradient

    def langevin_sampling(self, data, z_batch):
        for i in range(self.langevin_steps):
            if z_batch.grad is not None:
                z_batch.grad.zero_()

            generated = self.generator(z_batch)
            loss = self.loss_function(data, generated)
            loss.backward()
            # Update Z
            u = torch.randn(z_batch.shape[0], z_batch.shape[1]).to(self.device)
            z_batch = z_batch - (self.langevin_stepsize ** 2) / 2 * (z_batch.grad - self.latent_gradient(z_batch)) + self.langevin_stepsize * u
            z_batch = z_batch.detach().requires_grad_()
        return z_batch

    def train(self, train_loader, num_epochs, hidden_size, kernel_size,  channel_size, image_size, linear=False):
        if linear:
            self.generator = Linear_Generator(self.latent_size, hidden_size, channel_size, image_size).to(self.device)
        else:
            self.generator = Generator(self.latent_size, hidden_size, kernel_size, channel_size, image_size).to(self.device)
            self.init_weight()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)

        self.batch_size = train_loader.batch_size

        self.z = self.alpha * torch.randn(len(train_loader.dataset), self.latent_size) + (1 - self.alpha) * torch.mul(torch.randn(len(train_loader.dataset), self.latent_size), self.slab_variance)
        self.z = self.z.to(self.device).requires_grad_()

        for epochs in range(num_epochs):
            total_loss = 0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                z_batch = self.z[self.batch_size * batch_idx:self.batch_size * batch_idx + data.shape[0]].detach().requires_grad_()
                # Inferential Step
                z_batch = self.langevin_sampling(data, z_batch)
                self.z[self.batch_size * batch_idx:self.batch_size * batch_idx + data.shape[0]] = z_batch
                # Learning Step, update Generator
                optimizer.zero_grad()
                generated = self.generator(z_batch)
                loss = self.loss_function(data, generated)
                loss.backward()
                optimizer.step()
                # Sum up loss
                total_loss += float(loss)

            if epochs % 5 == 0 or epochs == num_epochs - 1:
                print("Interation:", epochs)
                print("Loss: ", total_loss)
                print("Average Z Sum:", torch.pow(self.z, 2).sum() / len(train_loader.dataset))
                ###
            else:
                print("Iteration:", epochs)
                print("Loss", total_loss)
                print("Average Z Sum:", torch.pow(self.z, 2).sum() / len(train_loader.dataset))

    def get_latent(self):
        return self.z
