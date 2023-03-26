import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, input_shape)

        self.z_mean = nn.Linear(32, latent_dim)
        self.z_log_var = nn.Linear(32, latent_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean,z_log_var,self.fc4(x)



class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, input_shape):


        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape

    def forward(self, x):

        z_mean, z_log_var,_ = self.encoder(x)
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std)
        z = eps * std + z_mean
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_log_var)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def loss_function(self, inputs, reconstructed, z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), axis=-1)
        reconstruction_loss = torch.sum(F.mse_loss(reconstructed, inputs, reduction='none'), axis=-1)
        vae_loss = torch.mean(reconstruction_loss + kl_loss)
        return vae_loss

def train_vae(vae, data, num_epochs=100, batch_size=32):
    optimizer = torch.optim.Adam(vae.parameters())
    #I should decrease the lr each 10 epochs due to the wierd ossilation in the loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_batches = data.shape[0] // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            batch_data = data[batch * batch_size:(batch + 1) * batch_size]
            batch_data = torch.tensor(batch_data, dtype=torch.float32)
            optimizer.zero_grad()
            reconstructed,z_mean,z_log_var = vae(batch_data)
            #z_mean, z_log_var, z = vae.encoder(batch_data)
            loss = vae.loss_function(batch_data, reconstructed,z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        scheduler.step() 





# Define the function to load the data
def load_data(path):
    # Load the time-series data from a file or other source
    data = np.loadtxt(path)
    return data

# Define the function to preprocess the data
def preprocess_data(data):
    # Normalize the data to be between 0 and 1
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

def plot_latent_space(latent_space):
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_space, latent_space)
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 1 in y')
    plt.title('Latent Space Visualization')
    plt.show()

def visualize_data(data,encoder):
    # Project the data into the latent space
    _,_,latent_space = encoder.forward(data)
    # Visualize the latent space
    plot_latent_space(latent_space)


# Define the function to sample points in the latent space
def sample_latent_space(num_samples, latent_dim):
    return np.random.normal(size=(num_samples, latent_dim))

# Generate new samples by sampling points in the latent space
def generate_latent(decoder,num_samples = 1000,latent_dim = 2):
    latent_points = sample_latent_space(num_samples, latent_dim)
    return decoder.predict(latent_points)

# Define the function to interpolate between two points in the latent space
def interpolate_latent_space(point_a, point_b, num_steps):
    return np.linspace(point_a, point_b, num_steps)

# Interpolate between two points in the latent space
def Interpolate(decoder,point_a = np.array([-2, -2]),point_b = np.array([2, 2]),num_steps = 10,):

    interpolation_points = interpolate_latent_space(point_a, point_b, num_steps)
    return decoder.forward(interpolation_points)



def Eval_vae(num_samples,latent_dim,decoder):
    # Define the function to evaluate the quality of the generated samples
    def evaluate_samples(samples):
        # Perform evaluation (e.g. using a classifier or other metric)
        #TODO: write the code
        return evaluation_results
    def satisfactory(evaluation_results):pass
        #TODO: write the code
    # Evaluate the quality of the generated samples
    evaluation_results = evaluate_samples(generated_samples)
    def refine_vae(vae):pass
    # Refine the VAE architecture and hyperparameters as necessary
    while not satisfactory(evaluation_results):
        vae = refine_vae(vae)
        generated_samples = decoder.predict(sample_latent_space(num_samples, latent_dim))
        evaluation_results = evaluate_samples(generated_samples)
