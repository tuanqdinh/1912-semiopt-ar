
"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""
from _init_ import *

from pixelcnnpp.model import *
from pixelcnnpp.utils import *

from vae.models.AutoEncoder import PointCloudAutoEncoder
from vae.EMD.emd import EmdDistance
from vae.ops import *

import warnings
warnings.filterwarnings("ignore")

# ==================Model======================
dim_mu = 20
beta = 3
obs = (1, 8, 8)
input_channels = obs[0]

netV = PointCloudAutoEncoder(size=1024, dim=3, name=args.model_name, enc_size=64,
		batch_size=args.batch_size).to(device)
netA = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
			input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)

# netV.apply(Helper.weights_init)
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

loss_fn = EmdDistance()
loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

# ==================Data======================
trainset=Provider.load_data(dataset_path, args.mode_space, space_dim, num_hiters,
							normalized = True, renew = args.flag_renew_data)
# [N, 32, 32, 3]
trainset = SFC.convert1dto2d(trainset)
trainset = torch.Tensor(trainset)
N = trainset.shape[0]
trainset = trainset.permute(0, 3, 2, 1).contiguous().view(N, 3, 1024)
dataloader = DataLoader(trainset, batch_size = args.batch_size,
						shuffle = True, drop_last = True)

# ==================Training======================
def sample(model, nsamples=2):
	model.train(False)
	data = torch.zeros(nsamples, obs[0], obs[1], obs[2]).to(device)
	for i in range(obs[1]):
		for j in range(obs[2]):
			with torch.no_grad():
				out = model(data, sample=True)
				out_sample = sample_op(out)
				data[:, :, i, j] = out_sample.data[:, :, i, j]
	return data

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


print('Load AE')
netV.load_state_dict(torch.load(netV_path))

start_time = time.time()
if os.path.isfile(netA_path) and args.flag_retrain:
	print('Load existing models')
	netA.load_state_dict(torch.load(netA_path))

total_step = len(trainset) // args.batch_size
for epoch in range(args.num_epochs):
	for batch_idx, batch_data in enumerate(dataloader):
		optimizerA.zero_grad()
		batch_data = batch_data.to(device)
		z = netV.encode(batch_data).detach()
		z = z.view(args.batch_size, 8, 8).unsqueeze(1)
		out_params = netA(z)
		loss = loss_op(z, out_params).mean()
		loss.backward()
		optimizerA.step()

		if 0 == batch_idx % args.log_step:
			log_loss(epoch, batch_idx, total_step, loss, start_time)
			start_time = time.time()

	print('save model A at epoch')
	torch.save(netA.state_dict(), netA_path)

print('sampling...')
with torch.no_grad():
	sample_z = sample(netA, 2).view(2, 64)
	sample_x = netV.decode(sample_z)
	plot_pc(sample_x, 0, name='final-generated')
