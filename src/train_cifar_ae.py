from _init_ import *
from vae.model_cifar import AE
import pdb

import warnings

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset loading
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model
    dim_embed = 128
    model = AE(dim_embed).cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)

    criterion = nn.MSELoss()

    total_step = len(trainset) // args.batch_size
    for epoch in range(args.num_epochs):
        train(args, epoch, model, criterion, train_loader, optimizer, total_step)
        test(args, epoch, model, criterion, test_loader)

    transf = transforms.Compose([UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    with torch.no_grad():
        model.eval()
        for batch_idx, (input, _) in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            
            raw_images = [transf(image.detach().cpu()) for image in input]
            inp = utils.make_grid(raw_images, nrow=4)
            
            output_images = [transf(image.detach().cpu()) for image in output]
            out = utils.make_grid(output_images, nrow=4)

            utils.save_image(inp, os.path.join(sample_path, 'inp.png'))
            utils.save_image(out, os.path.join(sample_path, 'out.png'))
            
            break
            
            
        
def train(args, epoch, model, criterion, train_loader, optimizer, total_step):
    start_time = time.time()
    model.train()
    for batch_idx, (input, _) in enumerate(train_loader):
        #pdb.set_trace()
        if batch_idx > 100 : break
        input = input.cuda()
        output = model(input)
        loss = criterion(input, output)
        if batch_idx % args.log_step == 0:
            log_loss(epoch, batch_idx, total_step, loss, start_time)
            start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(args, epoch, model, criterion, test_loader):
    loss_logger = AverageMeter()
    model.eval()
    for batch_idx, (input, _) in enumerate(test_loader):
        input = input.cuda()
        output = model(input)
        loss = criterion(input, output)
        loss_logger.update(loss.item())

    print('Test loss ', loss_logger.avg)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
if __name__ == "__main__":
    main()
    
