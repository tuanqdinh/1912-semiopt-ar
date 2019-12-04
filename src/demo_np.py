"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

from models.neural_process import NeuralProcess
from _init_demo import *
from np.np_utils import *

# ==================Model======================
# Define neural process for functions...
npnet = NeuralProcess(x_dim=1, y_dim=1, r_dim=10, z_dim=10, h_dim=10).to(device)
optimizer = optim.Adam(npnet.parameters(), lr=3e-4)
# 128 in total
num_context_range=(10, 50)
num_extra_target_range=(5, 10)

# ==================Training======================
def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
	var_p = torch.exp(logvar_p)
	kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
			 - 1.0 \
			 + logvar_p - logvar_q
	kl_div = 0.5 * kl_div.sum()
	return kl_div


def kl_divergence(p, q):
	# KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
	KLD = torch.distributions.kl.kl_divergence(p, q)
	return KLD


def np_loss(p_y_pred, y_target, q_target, q_context):
	"""
	Computes Neural Process loss.

	Parameters
	----------
	p_y_pred : one of torch.distributions.Distribution
		Distribution over y output by Neural Process.

	y_target : torch.Tensor
		Shape (batch_size, num_target, y_dim)

	q_target : one of torch.distributions.Distribution
		Latent distribution for target points.

	q_context : one of torch.distributions.Distribution
		Latent distribution for context points.
	"""
	# Log likelihood has shape (batch_size, num_target, y_dim). Take mean
	# over batch and sum over number of targets and dimensions of y
	log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
	# KL has shape (batch_size, r_dim). Take mean over batch and sum over
	# r_dim (since r_dim is dimension of normal distribution)
	kl = kl_divergence(q_target, q_context).mean(dim=0).sum()

	return -log_likelihood + kl


start_time = time.time()
if os.path.isfile(npnet_path) and args.flag_retrain:
	print('Load existing models')
	npnet.load_state_dict(torch.load(npnet_path))

data_iter = iter(dataloader)
total_step = len(data_iter)
x = torch.tensor(np.arange(args.signal_size)/100).unsqueeze(0).expand(args.batch_size, args.signal_size).unsqueeze(-1)
x = x.type(torch.FloatTensor).to(device)

lossfs = []
for epoch in range(args.num_epochs):
	# TODO:
	for batch_idx, batch_data in enumerate(dataloader):
		inputs = batch_data.to(device)

		# Sample number of context and target points
		num_context = np.random.randint(*num_context_range)
		num_extra_target = np.random.randint(*num_extra_target_range)

		# Create context and target points and apply neural process
		y = batch_data.unsqueeze(-1)
		y = y.type(torch.FloatTensor).to(device)

		x_context, y_context, x_target, y_target = \
			context_target_split(x, y, num_context, num_extra_target)
		p_y_pred, q_target, q_context = \
			npnet(x_context, y_context, x_target, y_target)

		loss = np_loss(p_y_pred, y_target, q_target, q_context)
		# TODO:
		# emd = Helper.emd(seqs, data)
		# cost = -l + emd
		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		lossf = loss.data.item()
		lossfs.append(lossf)

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()

	print("epoch average loss: {:.4f}".format(np.mean(lossfs)))
	print('save models at epoch')
	torch.save(npnet.state_dict(), npnet_path)
