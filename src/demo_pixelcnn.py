
"""
	@author: Tuan Dinh (tuandinh@cs.wisc.edu)
	@date: 08/14/2019
	@version:
	objective: train a PixelCNN model by using semi-discrete optimal transport loss in the replacement for the negative likelihood loss
"""

"""
	Notes (Debugging Mode):
	- sampling takes a lot of time => change to VAE instead
	or increase the batch size
"""

from pixelcnnpp.model import *
from pixelcnnpp.utils import *
from utils.semi_loss import semi_opt
# load all common variables and constants
from init import *

# ==================Model======================
obs = (1, 28, 28) # (C, H, W)
input_channels = obs[0] # number of channels
out_dim = obs[0] * args.nr_logistic_mix * 3 # number of outputs, 3 is for 3 parameters of a logistic distribution

# PixelCNN network
net = PixelCNN(nr_resnet=args.nr_resnet,
				nr_filters=args.nr_filters,
				input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)
net.apply(Helper.weights_init)

# optimization with cyclic scheduler
optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.9))
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr = 0.01, gamma=1.0, cycle_momentum=False)

# loss functions: 1d is for 1-channel image MNIST
loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
# sampling function
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)
# estimate density of a point x from given params of distribution
density_op = lambda x, params: discretized_mix_logistic_density_1d(x, params)

# path to store the trained model
net_path = os.path.join(model_path, 'net.pth')

# ==================Data======================
# scale data into [-1, 1] which is the support of logistic distribution
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
# load dataset with transforms
dataloader = torch.utils.data.DataLoader(
						datasets.MNIST(dataset_path, download=True, train=True,  transform=ds_transforms),
							batch_size=args.batch_size,
                            shuffle=True,
							**kwargs)
# test_loader  = torch.utils.data.DataLoader(datasets.MNIST(dataset_path, train=False,
                # transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

"""
	Func: generate samples by sequentially generating each pixel
	@params:
		model: trained PixelCNN
		obs: [C, H, W] of images
		sample_batch_size: number of samples
		training: True if in the training mode, False if in the testing mode
	@return:
		data: output images
		out_params: parameters of distributions of output images
"""
def generate_sample(model, obs, sample_batch_size=10, training=False):
	model.train(training)
	# outcome samples
	data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2], requires_grad=True).to(device)

	out_dim = obs[0] * args.nr_logistic_mix * 3 # number of outputs, 3 is for 3 parameters of a logistic distribution
	# set of all parameters of distributions of pixels
	out_params = torch.zeros(sample_batch_size, out_dim, obs[1], obs[2], requires_grad=True).to(device)
	# loop through each pixel
	for i in range(obs[1]):
		for j in range(obs[2]):
			data_v = data
			# params of all pixels
			out = model(data_v, sample=True)
			# use only the current position
			out_params[:, :, i, j] = out.data[:, :, i, j]
			# sampling
			out_sample = sample_op(out)
			# update the current pixel only
			data[:, :, i, j] = out_sample.data[:, :, i, j]

	return data, out_params

# ==================Training======================
# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
start_time = time.time()
# load the existed model if flag_retrain is True
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))

# for printing total_step only
data_iter = iter(dataloader)
total_step = len(data_iter)

"""
	Flow: we measure the divergence between 2 distributions: (1) (continuous) learned paramterized-net distributions and (2) (discrete) sample distribution of image
	- discretize continuous distribution by generating a num_source_samples number of samples
	- semi_opt is the semi-loss in the paper
"""
num_source_samples = 2 * args.batch_size #number of samples representing source distributions
for epoch in range(args.num_epochs):
	lossfs = [] # list of all loss values by epoch
	net.train(True)
	torch.cuda.synchronize()
	for batch_idx, (batch_data, _) in enumerate(dataloader):
		optimizer.zero_grad()
		batch_data = batch_data.to(device) # [N, C, H, W]
		# out_params = net(batch_data)
		# nll = loss_op(batch_data, out_params).mean()

		# generate samples from source distribution
		# sample_t: []
		# out_params: []
		sample_t, out_params = generate_sample(net, obs,  sample_batch_size=num_source_samples, training=True)

		# density of sampled data: to overcome the precision of floating point problem, we first estimate the log of density, then exp them
		# log_px: [N, 784]
		log_px = density_op(sample_t, out_params).sum(dim=[1, 2])
		px = torch.exp(log_px)
		# calculate semi loss: []
		loss = semi_opt(batch_data, sample_t, px, loss_fn)

		loss.backward()
		lossfs.append(loss.data.item())
		optimizer.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()
			print('loss: {:.4f}'.format(np.mean(lossfs)))

	# store  model
	print('save models at epoch')
	torch.save(net.state_dict(), net_path)
	# write to tensorboard
	writer.add_scalar('train/loss', np.mean(lossfs), epoch)
	# decrease learning rate
	scheduler.step()

	# generate samples for testing
	if 0 == (epoch + 1) % args.save_step:
		print('sampling')
		torch.cuda.synchronize()
		net.eval()
		# sample_t: []
		sample_t = generate_sample(net, obs, sample_batch_size=2)
		# rescale samples to the original range
		sample_t = rescaling_inv(sample_t)
		# pick 2 images for saving
		real_t = rescaling_inv(batch_data[:2, :, :, :])
		torchvision.utils.save_image(sample_t,
					os.path.join(sample_path, 'sample_{}.png'.format(epoch)),
                	nrow=5, padding=0)
writer.close()
