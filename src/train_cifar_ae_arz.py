from _init_ import *

from pixelcnnpp.model import *
from pixelcnnpp.utils import *

from vae.ops import *
from vae.model_cifar import AE
import pdb

import warnings
warnings.filterwarnings("ignore")

# ==================Model======================
dim_mu = 512
dim_embed = 128
obs = (1, 8, 16)
input_channels = obs[0]

netV = AE(dim_embed).cuda()
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

#netV_path = os.path.join(model_path, 'vnet.pth')
netV_path = os.path.join(model_path, '../../AE-CIFAR10/snapshots/ae_path.pth')
netA_path = os.path.join(model_path, 'anet.pth')

loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

# ==================Data======================
transform = transforms.Compose([transforms.ToTensor(),])
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)


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
        raw_images = [image.detach().cpu() for image in samples]
        inp = utils.make_grid(raw_images, nrow=4)
        print(sample_path)
        utils.save_image(inp, os.path.join(sample_path, name + '.png'))

print('Load AE')
netV.load_state_dict(torch.load(netV_path))

start_time = time.time()
if os.path.isfile(netA_path) and args.flag_retrain:
    print('Load existing models')
    netA.load_state_dict(torch.load(netA_path))

total_step = len(trainset) // args.batch_size
for epoch in range(args.num_epochs):
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        #if batch_idx > 100: break
        optimizerA.zero_grad()
        batch_data = batch_data.to(device)
        z = netV.encode(batch_data).detach()
        #pdb.set_trace()
        z = z.view(-1, 8, 16).unsqueeze(1)
        try:
            out_params = netA(z)
        except:
            pdb.set_trace()
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
    sample_z = sample(netA, 8).view(8, dim_embed)
    sample_x = netV.decode(sample_z)
    plot_pc(sample_x, 0, name='final-generated')
print('Done!')
