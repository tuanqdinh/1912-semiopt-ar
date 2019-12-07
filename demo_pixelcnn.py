
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from pixelcnnpp.model import *
from pixelcnnpp.utils import *
from utils.semi_loss import semi_opt

from _init_ import *

# ==================Model======================
obs = (1, 28, 28)
input_channels = obs[0]
out_dim = obs[0] * args.nr_logistic_mix * 3

net = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
			input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)
net.apply(Helper.weights_init)

optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.9))
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr = 0.01, gamma=1.0, cycle_momentum=False)

loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)
density_op = lambda x, params: discretized_mix_logistic_density_1d(x, params)

net_path = os.path.join(model_path, 'net.pth')

# ==================Data======================
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
dataloader = torch.utils.data.DataLoader(datasets.MNIST(dataset_path, download=True,
                        train=True, transform=ds_transforms), batch_size=args.batch_size,
                            shuffle=True, **kwargs)

# test_loader  = torch.utils.data.DataLoader(datasets.MNIST(dataset_path, train=False,
                # transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

def generate_sample(model, obs, sample_batch_size=10, training=False):
	model.train(training)
	data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2], requires_grad=True).to(device)
	out_params = torch.zeros(sample_batch_size, out_dim , obs[1], obs[2], requires_grad=True).to(device)

	for i in range(obs[1]):
		for j in range(obs[2]):
			data_v = data
			# params of pixels
			out = model(data_v, sample=True)
			out_params[:, :, i, j] = out.data[:, :, i, j]
			# sampling
			out_sample = sample_op(out)
			data[:, :, i, j] = out_sample.data[:, :, i, j]
	return data, out_params

# ==================Training======================
# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
start_time = time.time()
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))

data_iter = iter(dataloader)
total_step = len(data_iter)
"""
	sampling takes a lot of time => change to VAE instead
	or increase the batch size
"""
n_source = 2 * args.batch_size # args.batch_size * 2
for epoch in range(args.num_epochs):
	lossfs = []
	net.train(True)
	torch.cuda.synchronize()
	for batch_idx, (batch_data, _) in enumerate(dataloader):
		optimizer.zero_grad()
		batch_data = batch_data.to(device)
		# out_params = net(batch_data)
		# nll = loss_op(batch_data, out_params).mean()
		sample_t, out_params = generate_sample(net, obs, sample_batch_size=n_source, training=True)
		# density of sampled data
		# [N, 784]
		log_px = density_op(sample_t, out_params).sum(dim=[1, 2])
		px = torch.exp(log_px)
		loss = semi_opt(batch_data, sample_t, px)

		loss.backward()
		lossfs.append(loss.data.item())
		optimizer.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()
			print('loss: {:.4f}'.format(np.mean(lossfs)))

	### Saving and Sampling
	print('save models at epoch')
	torch.save(net.state_dict(), net_path)
	writer.add_scalar('train/loss', np.mean(lossfs), epoch)
	# decrease learning rate
	scheduler.step()

	if 0 == (epoch + 1) % args.save_step:
		print('sampling')
		torch.cuda.synchronize()
		net.eval()
		sample_t = generate_sample(net, obs, sample_batch_size=2)
		sample_t = rescaling_inv(sample_t)
		real_t = rescaling_inv(batch_data[:2, :, :, :])
		torchvision.utils.save_image(sample_t, os.path.join(sample_path, 'sample_{}.png'.format(epoch)),
                nrow=5, padding=0)
writer.close()
