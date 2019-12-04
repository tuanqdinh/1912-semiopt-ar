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

num_mixtures = 10
out_dim = num_mixtures * 3 * in_dim

net = MADE(in_dim, hidden_list, out_dim,
		   num_masks=args.num_masks).to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)

data_iter = iter(dataloader)
total_step = len(data_iter)
###################Training###################
def run_made_epoch(split, start_time, upto=None):
	# enable/disable grad for efficiency of forwarding test batches
	torch.set_grad_enabled(split == 'train')
	net.train() if split == 'train' else net.eval()
	nsamples = 1 if split == 'train' else 10
	lossfs = []

	for batch_idx, batch_data in enumerate(dataloader):
		# [B, 1024]
		batch_data = batch_data.to(device)
		for s in range(nsamples):
			# perform order/connectivity-agnostic training by resampling the masks
			if batch_idx % args.resample_every == 0 or split == 'test':  # if in test, cycle masks every time
				net.update_masks()
			# forward the model
			xbhat_new = net(batch_data)
			if s == 0:
				xbhat = xbhat_new
			else:
				xbhat += xbhat_new

		xbhat /= nsamples

		# evaluate the binary cross entropy loss
		params = xbhat.view(args.batch_size, in_dim, num_mixtures * 3)
		loss = discretized_mix_logistic_loss_1d(batch_data, params)
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
				print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
				# writer
				n_iter = batch_idx + epoch * total_step
				writer.add_scalar('Loss/train', np.mean(lossfs), n_iter)

		if split == 'test':
			print('Genearting samples')
			params = xbhat.view(args.batch_size, in_dim, num_mixtures * 3)
			samples = sample_from_discretized_mix_logistic_1d(params, nr_mix=num_mixtures)
			samples = (samples + 1) /2
			batch_data = (batch_data + 1)/2
			xs = SFC.decode_sfc(samples.cpu().numpy())
			rs = SFC.decode_sfc(batch_data.cpu().numpy())
			for idx in range(5):
				Plotter.plot_pc(xs[idx, :, :], sample_path, idx, name='sample')
				Plotter.plot_pc(rs[idx, :, :], sample_path, idx, name='real')

			if batch_idx > 0:
				print('Finnish')
				break


start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))

for epoch in range(args.num_epochs):
	run_made_epoch('train', start_time)

print('save models')
torch.save(net.state_dict(), net_path)
writer.close()

with torch.no_grad():
	run_made_epoch('test', start_time, upto=5)
