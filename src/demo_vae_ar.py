
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""
from made.model import MADE
from vae.model1d import VAE
from made.loss import discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d
from vae.ops import *
from gan.ops import log_loss_gan
from _init_demo import *

# ==================Model======================
# MODEL-MADE
hidden_list = list(map(int, args.hiddens.split(',')))
num_mixtures = 10
out_dim = num_mixtures * 3 * in_dim
# MODEL-VAE
dim_mu = 20

netV = VAE(dim_input=in_dim, dim_mu=dim_mu).to(device)
netA = MADE(in_dim, hidden_list, out_dim,
			num_masks=args.num_masks).to(device)

netA.apply(Helper.weights_init)
netV.apply(Helper.weights_init)

optimizerV = optim.Adam(netV.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizerA = optim.SGD(netA.parameters(), lr=args.lr)

netV_path = os.path.join(model_path, 'vnet.pth')
netA_path = os.path.join(model_path, 'anet.pth')

# ==================Training======================


def trainA(split, batch_data, batch_idx, upto=None):
	# enable/disable grad for efficiency of forwarding test batches
	nsamples = 1 if split == 'train' else 10
	for s in range(nsamples):
		# perform order/connectivity-agnostic training by resampling the masks
		if batch_idx % args.resample_every == 0 or split == 'test':  # if in test, cycle masks every time
			netA.update_masks()
		# forward the model
		xbhat_new = netA(batch_data)
		if s == 0:
			xbhat = xbhat_new
		else:
			xbhat += xbhat_new

	xbhat /= nsamples

	# evaluate the binary cross entropy loss
	params = xbhat.view(args.batch_size, in_dim, num_mixtures * 3)
	return params


start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	netV.load_state_dict(torch.load(netV_path))
	netA.load_state_dict(torch.load(netA_path))

for epoch in range(args.num_epochs):
	step = 0
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	while step < total_step:
		# (1) Update D network
		Helper.activate(netA)
		stepA = 0
		while stepA < args.critic_steps and step < total_step - 1:
			optimizerA.zero_grad()
			# real-data
			real_data = next(data_iter).to(device)
			netA.train()
			real_params = trainA('train', real_data, step)
			real_neg_likelihood, real_emd_distance = discretized_mix_logistic_loss_1d(
				real_data, real_params, emd=True)

			# fake-data - zero out
			with torch.no_grad():
				noise = torch.randn(args.batch_size, dim_mu).to(device)
				fake_data = netV.decode(noise)
			netA.train()
			fake_params = trainA('train', fake_data, step)
			fake_neg_likelihood, fake_emd_distance = discretized_mix_logistic_loss_1d(
				fake_data, fake_params, emd=True)

			# cost
			costA = real_neg_likelihood - fake_neg_likelihood
			costA.backward()
			optimizerA.step()

			stepA += 1
			step += 1

		# (2) Update V network
		Helper.deactivate(netA)
		optimizerV.zero_grad()
		netV.train()
		batch_data = next(data_iter).to(device)
		recon_batch, mu, logvar = netV(batch_data)
		recon, KLD = loss_function(recon_batch, batch_data, mu, logvar)
		netA.eval()
		v_params = trainA('test', recon_batch, step)
		v_neg_likelihood, v_emd_distance = discretized_mix_logistic_loss_1d(
			recon_batch, v_params, emd=True)

		costV = v_neg_likelihood + recon + KLD
		# backward
		costV.backward()
		optimizerV.step()

		# Print log info
		if total_step -1 == step:
			log_loss_gan(logf, epoch, args.num_epochs, step,
						 total_step, costA, costV, start_time)
			print("AR: {:.4f} - {:.4f}".format(real_neg_likelihood.data.cpu(), fake_neg_likelihood.data.cpu()))
			start_time = time.time()

			# sampling
			with torch.no_grad():
				noise = torch.randn(2, dim_mu).to(device)
				samples = netV.decode(noise)
			samples = (samples + 1) /2
			xs = SFC.decode_sfc(samples.cpu().data.numpy(), M=space_dim, p=num_hiters)
			# plot
			img = xs[0, :, :]
			filepath = sample_path + '/sample_{}_{}.png'.format(epoch, step)
			Plotter.plot_pc(img, filepath)

		step += 1

	print('save models at epoch')
	torch.save(netA.state_dict(), netA_path)
	torch.save(netV.state_dict(), netV_path)
