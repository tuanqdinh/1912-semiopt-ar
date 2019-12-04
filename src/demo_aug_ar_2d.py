
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""



from _init_ import *

from pixelcnnpp.model_latent import PixelCNN
from pixelcnnpp.utils import *

from vae.model_cifar import VAE
from vae.ops import *

import warnings
warnings.filterwarnings("ignore")

# ==================Model======================
dim_mu = 512
beta = 3
obs = (3, 32, 32)
input_channels = obs[0]

netV = VAE().to(device)
netA = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
			input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)

netV.apply(Helper.weights_init)
netA.apply(Helper.weights_init)
clip_value = 1.0

# for p in netA.parameters():
#     p.register_hook(lambda x: torch.clamp(x, -clip_value, clip_value))
# for p in netV.parameters():
#     p.register_hook(lambda x: torch.clamp(x, -clip_value, clip_value))

optimizerV = optim.Adam(netV.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizerA = optim.Adam(netA.parameters(), lr=args.lr, betas=(0.5, 0.9))

# schedulerV = lr_scheduler.StepLR(optimizerV, step_size=1, gamma=args.lr_decay)
# schedulerA = lr_scheduler.StepLR(optimizerA, step_size=1, gamma=args.lr_decay)

netV_path = os.path.join(model_path, 'vnet.pth')
netA_path = os.path.join(model_path, 'anet.pth')

loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

# ==================Data======================
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
trainset=Provider.load_data(dataset_path, args.mode_space, space_dim, num_hiters,
							normalized = True, renew = args.flag_renew_data)
# [N, 1024, 3] map to [-1, 1]
trainset = rescaling(trainset)
# [N, 32, 32, 3]
trainset = SFC.convert1dto2d(trainset)
# reshape [N, 3, 32, 32]
trainset = torch.Tensor(trainset)
trainset = trainset.permute(0, 3, 2, 1)
dataloader=DataLoader(trainset, batch_size = args.batch_size,
						shuffle = True, drop_last = True)

# ==================Training======================
def sample(model, latent, nsamples=2):
	model.train(False)
	data = torch.zeros(nsamples, obs[0], obs[1], obs[2]).to(device)
	for i in range(obs[1]):
		for j in range(obs[2]):
			with torch.no_grad():
				out = model(data, latent, sample=True)
				out_sample = sample_op(out)
				data[:, :, i, j] = out_sample.data[:, :, i, j]
	return data


start_time = time.time()
if os.path.isfile(netV_path) and args.flag_retrain:
	print('Load existing models')
	netV.load_state_dict(torch.load(netV_path))
	netA.load_state_dict(torch.load(netA_path))


data_iter = iter(dataloader)
total_step = len(data_iter)
for epoch in range(args.num_epochs):
	netA.train(True)
	torch.cuda.synchronize()
	netA.train()
	for step, imgs in enumerate(dataloader):
		real_data = imgs.to(device)

		vae_data = rescaling_inv(real_data)
		(latent_batch, recon_batch), mu, logvar = netV(vae_data, latent=True)
		BCE = recon_loss(recon_batch, vae_data)
		KLD = kld(mu, logvar)

		# (2) Update V network
		optimizerV.zero_grad()
		costV = BCE + KLD
		costV.backward()
		# Helper.print_gradnorm(netV)
		# nn.utils.clip_grad_norm_(netV.parameters(), clip_value)
		optimizerV.step()

		real_params = netA(real_data, latent_batch.detach())
		real_neg_likelihood = loss_op(real_data, real_params).mean()

		# (1) Update D network
		optimizerA.zero_grad()
		costA = real_neg_likelihood
		costA.backward()
		optimizerA.step()

		# Print log info
		if 0 == step % args.log_step:
			log_loss2(epoch, step,
					  total_step, costA, costV, start_time)
			print('BCE: {:.4f} KLD: {:.4f}'.format(BCE.data.cpu(), KLD.data.cpu()))
			start_time = time.time()
		# sampling
		if  0 == (step + 1) % args.save_step:
			print('sampling...')
			with torch.no_grad():
				n =2
				real_samples = rescaling_inv(real_data[:n])
				recon_samples = recon_batch[:n].view(n, obs[0], obs[1], obs[2])
				plot_samples(recon_samples, epoch, name='vae_recon{}'.format(step))
				plot_samples(real_samples, epoch, name='real{}'.format(step))

				sample_t = sample(netA, latent_batch[:n], n)
				sample_t = rescaling_inv(sample_t)
				plot_samples(sample_t, epoch, name='ar_recon{}'.format(step))

			# step increases
		step += 1

	# decrease learning rate
	# schedulerA.step()
	# schedulerV.step()
	# torch.cuda.synchronize()

	print('save models at epoch')
	torch.save(netA.state_dict(), netA_path)
	torch.save(netV.state_dict(), netV_path)

	writer.add_scalar('Loss/ar_total', costA.data.cpu(), epoch)
	writer.add_scalar('Loss/vae_total', costV.data.cpu(), epoch)

	if epoch > -1:
		with torch.no_grad():
			n = 4
			noise = torch.randn(n, dim_mu).to(device)
			latents, vae_samples = netV.decode(noise, True)
			latents = latents[:n]
			vae_samples = vae_samples[:n]
			sample_t = sample(netA, latents, n)
			sample_t = rescaling_inv(sample_t)
			vae_samples = vae_samples.view(n, obs[0], obs[1], obs[2])
			plot_samples(vae_samples, epoch, name='vae_sample')
			plot_samples(sample_t, epoch, name='ar_sample')

writer.close()
