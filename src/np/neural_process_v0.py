import numpy as np
import torch
import matplotlib.pyplot as plt
# matplotlib inline


class REncoder(nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 8

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, out_dim)

        self.a = torch.nn.ReLU()

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)

    def forward(self, inputs):
        return self.l2(self.a(self.l1(inputs)))


class ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the
    normally distributed function encoding, z."""

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.std1_size = out_dim

        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.std1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.std1.weight)

    def forward(self, inputs):
        softplus = torch.nn.Softplus()
        return self.m1(inputs), softplus(self.std1(inputs))


class Decoder(nn.Module):
    """
    Takes the x star points, along with a 'function encoding', z, and makes predictions.
    """

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(self.decoder, self).__init__()
        self.l1_size = 8
        self.l2_size = 8

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, out_dim)

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)

        self.a = torch.nn.Sigmoid()

    def forward(self, x_pred, z):
        """x_pred: No. of data points, by self.dim_x
        z: No. of samples, by self.dim_z
        """
        zs_reshaped = z.unsqueeze(-1).expand(
            z.shape[0], z.shape[1], x_pred.shape[0]).transpose(1, 2)
        xpred_reshaped = x_pred.unsqueeze(0).expand(
            z.shape[0], x_pred.shape[0], x_pred.shape[1])

        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=2)
        return self.l2(self.a(self.l1(xz))).squeeze(-1).transpose(0, 1), 0.005


def log_likelihood(mu, std, target):
    norm = torch.distributions.Normal(mu, std)
    return norm.log_prob(target).sum(dim=0).mean()


def KLD_gaussian(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16

    return (qs2 / ps2 + ((mu_q - mu_p)**2) / ps2 + torch.log(ps2 / qs2) - 1.0).sum() * 0.5


class NeuralProcess:
    def __init__(self, dims, n_z_samples, n_context):
        self.dim_x, self.dim_y, self.dim_r, self.dim_z = dims

        self.repr_encoder = REncoder(
            self.dim_x + self.dim_y, self.dim_r)  # (x,y)->r
        self.z_encoder = ZEncoder(self.dim_r, self.dim_z)  # r-> mu, std
        self.decoder = Decoder(self.dim_x + self.dim_z,
                               self.dim_y)  # (x*, z) -> y*
        self.n_context = n_context

        params = list(self.decoder.parameters()) + list(self.z_encoder.parameters())+
            list(self.repr_encoder.parameters())
        self.opt = optim.Adam(params, 1e-3)

    # Training
    def random_split_context_target(self, x, y):
        """Helper function to split randomly into context and target"""
        ind = np.arange(x.shape[0])
        ind_context = np.random.choice(ind, size=self.n_context, replace=False)
        ind_target = np.setdiff1d(ind, mask)
        # split data 
        x_context = x[mask]
        y_context = y[mask]
        x_target =  np.delete(x, mask, axis=0)
        y_target = np.delete(y, mask, axis=0)

        return

    def sample_z(self, mu, std, n):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(std.data.new(n, self.dim_z).normal_())
        return mu + std * eps

    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        xy = torch.cat([x, y], dim=1)
        rs = self.repr_encoder(xy)
        r_agg = rs.mean(dim=0)  # Average over samples
        return self.z_encoder(r_agg)  # Get mean and variance for q(z|...)


    def train(self, data, num_epochs, n_display=3000):
        all_x_np, all_y_np = data
        losses = []
        for t in range(num_epochs):
            # Generate data and process
            x_context, y_context, x_target, y_target = self.random_split_context_target(
                all_x_np, all_y_np, np.random.randint(1, 4))


    def run(self, x_c, x_t, y_c, y_t):

        # concatenate
        x_ct = torch.cat([x_c, x_t], dim=0)
        y_ct = torch.cat([y_c, y_t], dim=0)

        # Get latent variables for target and context, and for context only.
        z_mean_all, z_std_all = self.data_to_z_params(x_ct, y_ct)
        z_mean_context, z_std_context = self.data_to_z_params(x_c, y_c)
        # Sample a batch of zs using reparam trick.
        zs = self.sample_z(z_mean_all, z_std_all, self.n_z_samples)

        # Get the predictive distribution of y*
        mu, std = self.decoder(x_t, zs)

        # Compute loss and backprop
        loss = -log_likelihood(mu, std, y_t) + KLD_gaussian(z_mean_all, z_std_all, z_mean_context, z_std_context)
        losses.append(loss)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return losses
