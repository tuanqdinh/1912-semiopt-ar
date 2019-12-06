
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from pixelcnnpp.model import *
from pixelcnnpp.utils import *
from vae.ops import emd_batch, bce
from vae.EMD.emd import EmdDistance

from _init_ import *

# ==================Model======================
obs = (1, 28, 28)
input_channels = obs[0]

net = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
			input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)
net.apply(Helper.weights_init)

optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.9))
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr = 0.01, gamma=1.0, cycle_momentum=False)

loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

net_path = os.path.join(model_path, 'net.pth')

# ==================Data======================
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.dataset_path, download=True,
                        train=True, transform=ds_transforms), batch_size=args.batch_size,
                            shuffle=True, **kwargs)

test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
                transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

# [N, 1024, 3] map to [-1, 1]
trainset = rescaling(sorted_data)

dataloader=DataLoader(trainset, batch_size = args.batch_size,
                        shuffle = True, drop_last = True)

def generate_sample(model, obs, sample_batch_size=10):
	model.train(False)
	data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2]).to(device)
	for i in range(obs[1]):
		for j in range(obs[2]):
			with torch.no_grad():
				data_v = data
				out = model(data_v, sample=True)
				out_sample = sample_op(out)
				data[:, :, i, j] = out_sample.data[:, :, i, j]
	return data
# ==================Training======================
# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))
	if args.flag_plot:
		sample_t = generate_sample(net, obs, sample_batch_size=2)
		sample_t = rescaling_inv(sample_t)
		plot_samples(sample_t, -1, name='sample', nsamples=1)

data_iter = iter(dataloader)
total_step = len(data_iter)

if args.flag_demo:
	print('Plot-demo')
	samples = rescaling_inv(data_iter.next())
	samples = samples.permute(0, 3, 2, 1)
	samples = samples.contiguous().view(samples.shape[0], 1024, 3)
	x = samples.cpu().data.numpy()

	# x = sorted_data
	plot_step(x[0])

if args.flag_plot:
	print('Set up network')
	s = -7
	# [1, 3, 32, 32]
	data = data_iter.next()
	# data = torch.Tensor(np.arange(32*32*3)/10000).view([1, 3, 32, 32])
	# [1, 32, 32, 3]
	x = data.permute(0, 2, 3, 1)
	means = x.unsqueeze(-1).expand([1, 32, 32, 3, 10])
	# [1, 32, 32, ]
	log_scales = torch.zeros([1, 32, 32, 3, 10]) + s
	coeffs = torch.zeros([1, 32, 32, 3, 10])
	params = torch.cat([means, log_scales, coeffs], dim=-1).contiguous().view([1, 32, 32, 90])
	# logits-prob
	logit_probs = torch.ones([1, 32, 32, 10])
	params = torch.cat([logit_probs, params], dim=-1)
	print('sampling')
	params = params.permute(0, 3, 1, 2)
	sample_t = sample_op(params)
	sample_t = rescaling_inv(sample_t)
	real_t = rescaling_inv(data)
	plot_samples(sample_t, -1, name='sample-params')
	plot_samples(real_t, -1, name='real-params')
	print('EMD: ', emd_batch(data.to(device), params.to(device), loss_fn).mean().data.cpu())

for epoch in range(args.num_epochs):
	lossfs = []
	emds = []
	nlls = []
	net.train(True)
	torch.cuda.synchronize()
	for batch_idx, batch_data in enumerate(dataloader):
		optimizer.zero_grad()
		batch_data = batch_data.to(device)
		out_params = net(batch_data)
		nll = loss_op(batch_data, out_params).mean()
		emd = emd_batch(batch_data, out_params, loss_fn).mean()
		loss = nll + args.alpha * emd
		loss = emd
		loss.backward()
		lossfs.append(loss.data.item())
		nlls.append(nll.data.item())
		emds.append(emd.data.item())
		optimizer.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()
			print('nll: {:.4f} emd: {:.4f} loss: {:.4f}'.format(nll.data.cpu(), emd.data.cpu(), np.mean(lossfs)))

	### Saving and Sampling
	print('save models at epoch')
	torch.save(net.state_dict(), net_path)
	writer.add_scalar('train/loss', np.mean(lossfs), epoch)
	writer.add_scalar('train/emd', np.mean(emds), epoch)
	writer.add_scalar('train/nll', np.mean(nlls), epoch)
	# decrease learning rate
	scheduler.step()

	if 0 == (epoch + 1) % args.save_step:
		print('sampling')
		torch.cuda.synchronize()
		net.eval()
		sample_t = generate_sample(net, obs, sample_batch_size=2)
		sample_t = rescaling_inv(sample_t)
		real_t = rescaling_inv(batch_data[:2, :, :, :])
		plot_samples(sample_t, epoch, name='sample')
		plot_samples(real_t, epoch, name='real')
writer.close()
