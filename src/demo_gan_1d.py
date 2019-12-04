"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from gan.vgan import Generator, Discriminator
from gan.ops import *
from _init_demo import *

#### ==================Model======================
netG = Generator(args.embed_size, args.signal_size).to(device)
netD = Discriminator(args.signal_size).to(device)
netD.apply(Helper.weights_init)
netG.apply(Helper.weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9))

netG_path = os.path.join(model_path, 'netG.pth')
netD_path = os.path.join(model_path, 'netD.pth')

#### ==================Training======================
def trainD(batch_data, penalty=False):
	# data
	real_data = batch_data.to(device)
	with torch.no_grad():
		real_data_zero = autograd.Variable(real_data)
	# train real
	realD = netD(real_data_zero)

	# train with fake
	noise = torch.randn(real_data_zero.shape[0], args.embed_size).to(device)
	with torch.no_grad():
		noise_zero = autograd.Variable(noise)  # totally freeze netG
	fake_data = autograd.Variable(netG(noise_zero).data)
	#### This will prevent gradient on params of G
	fakeD = netD(fake_data)
	if penalty:
		# train with gradient penalty
		# gradient_penalty = get_gradient_penalty(netD, real_data_zero.data, fake_data.data)
		grad_penalty = get_gradient_penalty(netD, real_data_zero, fake_data)
	else:
		grad_penalty = None

	return realD, fakeD, grad_penalty

def trainG():
	noise = torch.randn(args.batch_size, args.embed_size).to(device)
	with torch.no_grad():
		noise_zero = autograd.Variable(noise)
	fake_data = netG(noise_zero)
	fakeD = netD(fake_data)
	return fakeD, fake_data

start_time = time.time()
if os.path.isfile(netG_path) and args.flag_retrain:
	print('Load existing models')
	netG.load_state_dict(torch.load(netG_path))
	netD.load_state_dict(torch.load(netD_path))
	if args.flag_plot:
		samples = generate_sample(netG, nsamples=10, embed_size=args.embed_size)
		samples = (samples + 1) /2
		xs = SFC.decode_sfc(samples.cpu().data.numpy(), M=2, p=10)
		for idx in range(2):
			Plotter.plot_pc2(xs[idx, :, :], sample_path, idx, name='sample')

		from IPython import embed; embed()

for epoch in range(args.num_epochs):
	step = 0
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	while step < total_step:
		# (1) Update D network
		Helper.activate(netD)
		stepD = 0
		while stepD < args.critic_steps and step < total_step - 1:
			optimizerD.zero_grad()
			realD, fakeD, grad_penalty = trainD(next(data_iter), penalty=True)
			realD = realD.mean()
			fakeD = fakeD.mean()
			# cost
			costD = fakeD - realD + args.lam * grad_penalty
			costD.backward()
			optimizerD.step()
			stepD += 1
			step += 1
		# (2) Update G network
		Helper.deactivate(netD)
		optimizerG.zero_grad()
		fakeD, fake_data = trainG()
		fakeD = fakeD.mean()
		costG = -fakeD
		# backward
		costG.backward()
		optimizerG.step()

		# Print log info
		if 0 == step % args.log_step:
			log_loss_gan(logf, epoch, args.num_epochs, step, total_step, costD, costG, start_time)
			start_time = time.time()

		step += 1

	print('save models at epoch')
	torch.save(netG.state_dict(), netG_path)
	torch.save(netD.state_dict(), netD_path)
