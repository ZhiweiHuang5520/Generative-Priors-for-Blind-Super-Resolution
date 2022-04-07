import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import torch.nn as nn
from utils import *
from dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from loss import GradLoss, SSIM
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description='Process the SR network')

parser.add_argument('--experiment', default='train', help='The path to store sampled images and models' )
parser.add_argument('--continue_train', action='store_true', default= True, help='Continue training on the saved best model' )
parser.add_argument('--WithPriors', action='store_true', default= True, help='Use the model with generative priors' )
parser.add_argument('--sf', type=int, default=3, help='Scale factor' )
parser.add_argument('--dataset', type=str, default='DF2K', help='dataset: DF2K')
parser.add_argument('--val_root', type=str, default='val', help='Validate dataset')
parser.add_argument('--epochId', type=int, default=80, help='The number of epochs being trained')
parser.add_argument('--batch_size', type=int, default= 2, help='The size of a batch' )
parser.add_argument('--crop_size', type=int, default= 64, help='The size of a croped input of training image' )
parser.add_argument('--val_batch', type=int, default=1, help='Validate dataset batch size')
parser.add_argument('--val_crop', type=int, default= 64, help='The size of a croped input of validation image' )
parser.add_argument('--initLR', type=float, default=0.0001, help='The initial learning rate')
parser.add_argument('--multi_step_lr', type=int, nargs='+', default=[30,70,120], help='The epoch to decrease learning rate')
parser.add_argument('--up_ch', type=int, default=64, help='Channel number of upsampled input')
parser.add_argument('--num_block', type=int, default=23, help='The number of RRDB blocks')
parser.add_argument('--num_grow_ch', type=int, default=32, help='Channel number for each growth')
parser.add_argument('--num_feat', type=int, default=64, help='Channel number of intermediate features')
parser.add_argument('--num_out_ch', type=int, default=3, help='Channel number of intermediate features')
parser.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader')


def main(opt):
    # Initialize tensorboard

    writer = SummaryWriter('run/'+opt.experiment+'x'+str(opt.sf))


    # Initialize network
    if not opt.WithPriors:
        encoder = model.Encoder(up_ch = opt.up_ch, sf = opt.sf, num_block = opt.num_block, num_feat = opt.num_feat, num_grow_ch = opt.num_grow_ch)
        decoder = model.Decoder(up_ch = opt.up_ch, sf= opt.sf, num_feat = opt.num_feat, num_out_ch = opt.num_out_ch)
    else:
        encoder = model.PriorEncoder(up_ch = opt.up_ch, sf = opt.sf, num_block = opt.num_block, num_feat = opt.num_feat, num_grow_ch = opt.num_grow_ch, in_size = opt.crop_size)
        decoder = model.PriorDecoder(up_ch = opt.up_ch, sf= opt.sf, num_feat = opt.num_feat, num_out_ch = opt.num_out_ch)
    # Move network and containers to gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True if device == 'cuda' else False

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if opt.continue_train:#== False:
        print("@@@@@@@@@@@ Continue Training from Checkpoint")
        encoder_root = 'run/'+opt.experiment+'x'+str(opt.sf)+'/saved_model/best_encoder.pth'
        decoder_root = 'run/'+opt.experiment+'x'+str(opt.sf)+'/saved_model/best_decoder.pth'
        encoder.load_state_dict(torch.load(encoder_root))
        decoder.load_state_dict(torch.load(decoder_root))


    # Initialize dataLoader
    HRroot = 'datasets/{}/train/HR/'.format(opt.dataset)
    LRroot = 'datasets/{}/train/lr_x{}'.format(opt.dataset, opt.sf)

    trainDataset = MyDataset(HRroot, LRroot, opt.sf, opt.crop_size)
    trainDataloader = DataLoader(trainDataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.workers)

    valHRroot = 'datasets/{}/val/HR/'.format(opt.dataset)
    valLRroot = 'datasets/{}/val/lr_x{}'.format(opt.dataset, opt.sf)


    valDataset = MyDataset(valHRroot, valLRroot, opt.sf, opt.val_batch, 'val', opt.val_crop)
    valDataloader = DataLoader(valDataset, batch_size=opt.val_batch,
                                 shuffle=False, num_workers=opt.workers)


    # Initialize optimizer
    optimizerNet = optim.Adam([
        {'params': encoder.parameters(), 'lr': opt.initLR / 5},
        {'params': decoder.parameters()}
    ], lr=opt.initLR)
    if opt.multi_step_lr:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizerNet, opt.multi_step_lr)

    # loss function
    MSE = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    MGE = GradLoss().to(device)

    iteration = 0
    epoch = opt.epochId


    # save model
    Path('run/'+opt.experiment+'x'+str(opt.sf)+'/saved_model').mkdir(parents=True, exist_ok=True)

    best_ssim = 0
    for epoch in range(0, opt.epochId):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        running_mge = 0.0
        for i, (lrim, hrim) in enumerate(trainDataloader):
            iteration += 1

            optimizerNet.zero_grad()
            lrim = lrim.to(device)
            hrim = hrim.to(device)

            if not opt.WithPriors:
                f1, f2= encoder(lrim)
                sr_out = decoder(f1, f2)
            else:
                f1,f2,f3,f4 = encoder(lrim)
                sr_out = decoder(f1,f2,f3,f4)

            mse_loss = MSE(hrim, sr_out)
            ssim_value = ssim_loss(hrim * 0.5 + 0.5, sr_out * 0.5 + 0.5)
            mge_loss = MGE(hrim, sr_out)
            total_loss = mse_loss + (1-ssim_value) + mge_loss
            total_loss.backward()
            optimizerNet.step()

            running_loss += total_loss
            running_mse += mse_loss
            running_ssim += ssim_value
            running_mge += mge_loss

            if i % 10 == 9:  # every 10 mini-batches...
                total_iter = epoch * len(trainDataset) + i*opt.batch_size
                print(f"loss: {running_loss.item()*10:>7f}  Epoch:{epoch:>d}  Curr Iter: [{ i+1:>5d}/{len(trainDataloader):>5d}]")
                # ...log the running loss
                writer.add_scalar('training mixed loss',
                                  running_loss / 10,
                                  total_iter)

                writer.add_scalar('training MSE loss',
                                  running_mse / 10,
                                  total_iter)

                writer.add_scalar('training SSIM',
                                  running_ssim / 10,
                                  total_iter)

                writer.add_scalar('training MGE loss',
                                  running_mge / 10,
                                  total_iter)

                writer.add_figure('LR input vs. SR output',
                                  plot_versus((((lrim[1] * 0.5) + 0.5) * 255.0+1e-5),
                                              (((sr_out[1] * 0.5) + 0.5) * 255.0+1e-5),
                                              (((hrim[1] * 0.5) + 0.5) * 255.0+1e-5)),
                                              global_step= total_iter)

                running_loss = 0.0
                running_mse = 0.0
                running_ssim = 0.0
                running_mge = 0.0

        #evaluate
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_ssim = 0.0
        val_mge = 0.0
        with torch.no_grad():
            for (lrim, hrim) in valDataloader:
                lrim = lrim.to(device)
                hrim = hrim.to(device)

                if not opt.WithPriors:
                    f1, f2 = encoder(lrim)
                    sr_out = decoder(f1, f2)
                else:
                    f1, f2, f3, f4 = encoder(lrim)
                    sr_out = decoder(f1, f2, f3, f4)

                val_mse_loss = MSE(hrim, sr_out)
                val_ssim_value = ssim_loss(hrim * 0.5 + 0.5, sr_out * 0.5 + 0.5)
                val_mge_loss = MGE(hrim, sr_out)
                val_total_loss = val_mse_loss + (1 - val_ssim_value) + val_mge_loss

                val_loss += val_total_loss
                val_mse += val_mse_loss
                val_ssim += val_ssim_value
                val_mge += val_mge_loss

            print(f"Validation loss: {val_loss.item():>7f}  Epoch:{epoch:>d} ")
            # ...log the running loss
            len_val = len(valDataset) / opt.val_batch
            writer.add_scalar('validation mixed loss',
                              val_loss / len_val,
                              epoch)

            writer.add_scalar('validation MSE loss',
                              val_mse / len_val,
                              epoch)

            writer.add_scalar('validation SSIM',
                              val_ssim / len_val,
                              epoch)

            writer.add_scalar('validation MGE loss',
                              val_mge / len_val,
                              epoch)

            writer.add_figure('validation LR input vs. SR output vs. HR ',
                              plot_versus((((lrim[0]* 0.5) + 0.5) * 255.0),
                              (((sr_out[0]* 0.5) + 0.5) * 255.0),
                                          (((hrim[0]* 0.5) + 0.5) * 255.0)),
                              global_step=epoch)
            if best_ssim<val_ssim:
                torch.save(encoder.state_dict(), 'run/'+opt.experiment+'x'+str(opt.sf)+'/saved_model/best_encoder.pth')
                torch.save(decoder.state_dict(), 'run/'+opt.experiment+'x'+str(opt.sf)+'/saved_model/best_decoder.pth')
                best_ssim = val_ssim.item()

        lr_scheduler.step()

    writer.close()
    print('Finished Training')

if __name__=="__main__":
    # The detail network setting
    opt = parser.parse_args()
    print(opt)

    for i in [3]:
        opt.sf = i
        if i == 8:
            opt.batch_size = 128
            opt.crop_size = 8
            opt.val_crop = 64
        main(opt)
    sys.exit()
    




