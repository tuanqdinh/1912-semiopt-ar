
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""
from vae.model_ae import AE2
from vae.ops import *
from gan.ops import log_loss_gan
from vae.EMD.emd import EmdDistance
from _init_ import *
# ==================Model======================
# MODEL-VAE
dim_mu = 20

netV = AE2(dim_input=in_dim, device=device).to(device)
netV.apply(Helper.weights_init)
optimizerV = optim.Adam(netV.parameters(), lr=args.lr, betas=(0.5, 0.9))
netV_path = os.path.join(model_path, 'vnet.pth')

loss_fn = EmdDistance()

def plot_pc(samples, epoch, name, nsamples=1, color=False):
	# samples = samples.permute(0, 2, 1)
	# samples = samples.contiguous().view(samples.shape[0], 1024, 3)
	samples = samples.cpu().data.numpy()
	# plot
	for i in range(nsamples):
		img = samples[i, :, :]
		filepath = sample_path + '/{}_{}_{}.png'.format(epoch, i, name)
		if color:
			Plotter.plot_pc_color(img, filepath)
		else:
			Plotter.plot_pc(img, filepath)
# ==================Training======================

trainset = Provider.load_data(dataset_path, args.mode_space, space_dim, num_hiters,
							normalized = True, renew = args.flag_renew_data)
trainset = torch.Tensor(trainset)
trainset = trainset.permute(0, 2, 1)
dataloader = DataLoader(trainset, batch_size = args.batch_size,
						shuffle = True, drop_last = True)

start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	netV.load_state_dict(torch.load(netV_path))

for epoch in range(args.num_epochs):
	step = 0
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	lossfs = []
	for batch_idx, batch_data in enumerate(dataloader):
		optimizerV.zero_grad()
		batch_data = next(data_iter).to(device)
		recon_batch, parts = netV(batch_data)
		batch_data = batch_data.permute(0, 2, 1)
		# [N, 90, 32, 32]

		loss1 = loss_fn(batch_data, recon_batch).mean()
		loss2 = loss_fn(parts[0], parts[1]).mean()

		loss = loss1 - 0.01 * loss2

		lossfs.append(loss.data.item())
		loss.backward()
		optimizerV.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()
			print('loss: {:.4f}'.format(np.mean(lossfs)))

	# plot_pc(batch_data, epoch, name='real')
	plot_pc(recon_batch, epoch, name='sample')
	plot_pc(parts[0], epoch, name='sample_part1')
	plot_pc(parts[1], epoch, name='sample_part2')

	print('save models at epoch')
	torch.save(netV.state_dict(), netV_path)
