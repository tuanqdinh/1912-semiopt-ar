"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MNIST') # required
parser.add_argument('--model_name', type=str, default='pixel-semi') #required
parser.add_argument('--mode_space', type=int, default=3) #required
parser.add_argument('--result_path', type=str, default='../result', help='output path')
parser.add_argument('--data_path', type=str, default='../data', help='path for data')

# Training
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_size', type=int, default=16)
parser.add_argument('--signal_size', type=int, default=128)
parser.add_argument('--critic_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=50, help='step size for saving trained models')

parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')
parser.add_argument('--flag_renew_data', default=False, action='store_true', help='Reuse dataset')
parser.add_argument('--flag_demo', default=False, action='store_true', help='Plot')


### WaveNet
parser.add_argument('--layer_size', type=int, default=10,
                    help='layer_size: 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]')
parser.add_argument('--stack_size', type=int, default=5,
                    help='stack_size: 5 = stack[layer1, layer2, layer3, layer4, layer5]')
parser.add_argument('--in_channels', type=int, default=1,
                    help='input channel size. mu-law encode factor, one-hot size')
parser.add_argument('--res_channels', type=int, default=1, help='number of channel for residual network')

#### hilbert

#### made
# MADE
parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")

### PixelCNN
parser.add_argument('--nr_resnet', type=int, default=4,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('--nr_filters', type=int, default=50,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')

# Model parameters
args = parser.parse_args()
# print(args)
