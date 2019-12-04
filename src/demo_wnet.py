"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

from models.wnet import WaveNet
from _init_demo import *

# ==================Model======================
wnet = WaveNet(args.layer_size, args.stack_size,
			   args.in_channels, args.res_channels).to(device)

dataloader = DataLoader(args.data_dir, wnet.receptive_fields, args.in_channels)

criterion = nn.CrossEntropyLoss().to(device)
optimzier = optim.Adam(wnet.parameters(), lr=args.lr)

# ==================Training======================
start_time = time.time()
if os.path.isfile(wnet_path) and args.flag_retrain:
	print('Load existing models')
	wnet.load_state_dict(torch.load(wnet_path))

for epoch in range(args.num_epochs):
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	# TODO:
	for batch_idx, object in enumerate(dataloader):
		inputs = object.to(device)

		outputs = wnet(inputs)
		loss = criterion(outputs.view(-1, args.in_channels),
						 inputs.long().view(-1))
		# TODO:
		# emd = Helper.emd(seqs, data)
		# cost = -l + emd
		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Print log info
		if 0 == batch_idx % args.log_step:
			log_loss(epoch, step, total_step, loss, start_time)
			start_time = time.time()

	print('save models at epoch')
	torch.save(wnet.state_dict(), wnet_path)
