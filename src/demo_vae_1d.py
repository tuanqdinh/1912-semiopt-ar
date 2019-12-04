
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from vae.model1d import VAE
from vae.ops import *
from _init_demo import *

# ==================Model======================
dim_mu = 20
net = VAE(dim_input=in_dim, dim_mu=dim_mu).to(device)
net.apply(Helper.weights_init)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.9))

net_path = os.path.join(model_path, 'net.pth')

# ==================Training======================
start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))
	if args.flag_plot:
		with torch.no_grad():
			noise = torch.randn(2, dim_mu).to(device)
			samples = net.decode(noise)
		samples = (samples + 1) /2
		xs = SFC.decode_sfc(samples.cpu().data.numpy(), M=space_dim, p=num_hiters)
		# plot
		img = xs[0, :, :]
		filepath = sample_path + '/sample.png'
		Plotter.plot_pc(img, filepath)
		from IPython import embed; embed()

data_iter = iter(dataloader)
total_step = len(data_iter)
for epoch in range(args.num_epochs):
	lossfs = []
	net.train()
	for batch_idx, object in enumerate(dataloader):
		object = object.to(device)

		optimizer.zero_grad()
		recon_batch, mu, logvar = net(object)
		loss = loss_function(recon_batch, object, mu, logvar)
		loss.backward()
		lossf = loss.data.item()
		lossfs.append(lossf)
		optimizer.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()
			print("epoch average loss: {:.4f}".format(np.mean(lossfs)))

		if 0 == batch_idx % args.save_step:
			with torch.no_grad():
				noise = torch.randn(2, dim_mu).to(device)
				samples = net.decode(noise)
			samples = (samples + 1) /2
			xs = SFC.decode_sfc(samples.cpu().data.numpy(), M=space_dim, p=num_hiters)
			# plot
			img = xs[0, :, :]
			filepath = sample_path + '/sample_{}_{}.png'.format(epoch, batch_idx)
			Plotter.plot_pc(img, filepath)

	print('save models at epoch')
	torch.save(net.state_dict(), net_path)
