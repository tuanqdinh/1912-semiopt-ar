"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

from models.made import MADE
from _init_demo import *
from utils.loss import discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d

# ==================Model======================
# MODEL
hidden_list = list(map(int, args.hiddens.split(',')))

in_dim = in_dim //2
num_mixtures = 10
out_dim = num_mixtures * 3 * in_dim

net1 = MADE(in_dim, hidden_list, out_dim,
		   num_masks=args.num_masks).to(device)

net2 = MADE(in_dim, hidden_list, out_dim,
		   num_masks=args.num_masks).to(device)

net_params = list(net1.parameters()) + list(net2.parameters())
optimizer = optim.Adam(net_params, lr=args.lr)

data_iter = iter(dataloader)
total_step = len(data_iter)
###################Training###################
def run_made_epoch(split, start_time, upto=None):
	# enable/disable grad for efficiency of forwarding test batches
	torch.set_grad_enabled(split == 'train')
	net1.train() if split == 'train' else net1.eval()
	net2.train() if split == 'train' else net2.eval()
	nsamples = 1 if split == 'train' else 10
	lossfs = []

	for batch_idx, batch_data in enumerate(dataloader):
		# [B, 1024]
		batch_data = batch_data.to(device)
		for s in range(nsamples):
			# perform order/connectivity-agnostic training by resampling the masks
			if batch_idx % args.resample_every == 0 or split == 'test':  # if in test, cycle masks every time
				net1.update_masks()
				net2.update_masks()
			# forward the model
			batch_data_1 = batch_data[:, :in_dim]
			batch_data_2 = batch_data[:, in_dim:]
			xbhat_new1 = net1(batch_data_1)
			xbhat_new2 = net2(batch_data_2)
			xbhat_new = torch.cat([xbhat_new1, xbhat_new2], dim=1)
			if s == 0:
				xbhat = xbhat_new
			else:
				xbhat += xbhat_new

		xbhat /= nsamples

		# evaluate the binary cross entropy loss
		params = xbhat.view(args.batch_size, in_dim * 2, num_mixtures * 3)
		kl, emd_loss = discretized_mix_logistic_loss_1d(batch_data, params, emd=True)
		loss = 100 * emd_loss + kl
		lossf = loss.data.item()
		lossfs.append(lossf)

		# backward/update
		if split == 'train':
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if total_step-1 == batch_idx:
				log_loss(epoch, batch_idx, total_step, loss, start_time)
				start_time = time.time()
				print("EMD loss {:.4f}, KL {:.4f}".format(emd_loss.data.item(), kl.data.item()))
				print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
				# writer
				n_iter = batch_idx + epoch * total_step
				writer.add_scalar('Loss/train', np.mean(lossfs), n_iter)

		if split == 'test':
			print('Generating samples')
			params = xbhat.view(args.batch_size, in_dim * 2, num_mixtures * 3)
			samples = sample_from_discretized_mix_logistic_1d(params, nr_mix=num_mixtures)
			samples = (samples + 1) /2
			batch_data = (batch_data + 1)/2
			xs = SFC.decode_sfc(samples.cpu().numpy(), M=space_dim, p=num_hiters)
			rs = SFC.decode_sfc(batch_data.cpu().numpy(), M=space_dim, p=num_hiters)
			for idx in range(5):
				if args.mode_dataset > 1:
					Plotter.plot_pc2(xs[idx, :, :], sample_path, idx, name='sample')
					Plotter.plot_pc2(rs[idx, :, :], sample_path, idx, name='real')
				else:
					Plotter.plot_pc(xs[idx, :, :], sample_path, idx, name='sample')
					Plotter.plot_pc(rs[idx, :, :], sample_path, idx, name='real')

			if batch_idx > 0:
				print('Finnish')
				break


start_time = time.time()
if os.path.isfile(net1_path) and args.flag_retrain:
	print('Load existing models')
	net1.load_state_dict(torch.load(net1_path))
	net2.load_state_dict(torch.load(net2_path))


for epoch in range(args.num_epochs):
	run_made_epoch('train', start_time)
	print('save models')
	torch.save(net1.state_dict(), net1_path)
	torch.save(net2.state_dict(), net2_path)


writer.close()

with torch.no_grad():
	run_made_epoch('test', start_time, upto=5)
