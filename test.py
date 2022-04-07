import argparse
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dataset import MyDataset
from loss import GradLoss, SSIM
from piq import psnr
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser(description='Test the SR network')

parser.add_argument('--base_interpolate', action='store_true', default= True, help='Baseline Bicubic' )
parser.add_argument('--WithPriors', action='store_true', default= True, help='Use the model with generative priors' )
parser.add_argument('--sf', type=int, default=2, help='Scale factor')
parser.add_argument('--model_root', type=str, default='./run', help='Root of saved model weight')
parser.add_argument('--dataset', type=str, default='Set5', help='dataset: Set5, Set14, BSD100, Urban100')
parser.add_argument('--val_batch', type=int, default=1, help='Validate dataset batch size')
parser.add_argument('--val_crop', type=int, default= 64, help='The size of a croped input of validation image' )
parser.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader')
parser.add_argument('--up_ch', type=int, default=64, help='Channel number of upsampled input')
parser.add_argument('--num_block', type=int, default=23, help='The number of RRDB blocks')
parser.add_argument('--num_grow_ch', type=int, default=32, help='Channel number for each growth')
parser.add_argument('--num_feat', type=int, default=64, help='Channel number of intermediate features')
parser.add_argument('--num_out_ch', type=int, default=3, help='Channel number of intermediate features')

def main(opt):
    if opt.base_interpolate:
        with torch.no_grad():
            print("@@@@@@@@ bicubic @@@@@@@@")
            writer = SummaryWriter('test_res/' + 'x{}/baseline/{}'.format(opt.sf, opt.dataset))

            # Initialize dataLoader
            valHRroot = 'datasets/{}/test/HR/'.format(opt.dataset)
            valLRroot = 'datasets/{}/test/lr_x{}'.format(opt.dataset, opt.sf)

            valDataset = MyDataset(valHRroot, valLRroot, opt.sf, opt.val_batch, 'test', opt.val_crop)
            valDataloader = DataLoader(valDataset, batch_size=opt.val_batch,
                                       shuffle=True, num_workers=opt.workers)

            ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

            running_ssim = 0.0
            running_psnr = 0.0

            for i, (lrim, hrim) in enumerate(valDataloader):
                sr_out = F.interpolate(lrim, scale_factor=opt.sf, mode='bicubic')
                save_image(torch.clamp(sr_out * 0.5 + 0.5,0.0,1.0), 'test_res/' + 'x{}/baseline/{}'.format(opt.sf, opt.dataset)+str(i)+'.png')
                            #* 0.5) + 0.5)
                writer.add_figure('validation LR input vs. SR output vs. HR ',
                                  plot_versus((((lrim[0] * 0.5) + 0.5) * 255.0),
                                              (((sr_out[0] * 0.5) + 0.5) * 255.0),
                                              (((hrim[0] * 0.5) + 0.5) * 255.0)),
                                  global_step=i)

                ssim_value = ssim_loss(hrim * 0.5 + 0.5, sr_out * 0.5 + 0.5)
                topsnr_hr = torch.clamp(hrim * 0.5 + 0.5, min=0.0, max=1.0)
                topsnr_sr = torch.clamp(sr_out * 0.5 + 0.5, min=0.0, max=1.0)
                psnr_value = psnr(topsnr_hr * 0.5 + 0.5, topsnr_sr * 0.5 + 0.5)

                running_ssim += ssim_value
                running_psnr += psnr_value
            len_val = len(valDataset) * opt.val_batch
            print(f"Test SSIM: {running_ssim.item() / len_val:>7f}  Test PSNR: {running_psnr.item() / len_val:>7f}")




    writer = SummaryWriter('test_res/'+'x{}/{}'.format(opt.sf, opt.dataset))


    # Initialize network
    # Initialize network
    if not opt.WithPriors:
        encoder = model.Encoder(up_ch=opt.up_ch, sf=opt.sf, num_block=opt.num_block, num_feat=opt.num_feat,
                                num_grow_ch=opt.num_grow_ch)
        decoder = model.Decoder(up_ch=opt.up_ch, sf=opt.sf, num_feat=opt.num_feat, num_out_ch=opt.num_out_ch)
    else:
        encoder = model.PriorEncoder(up_ch=opt.up_ch, sf=opt.sf, num_block=opt.num_block, num_feat=opt.num_feat,
                                     num_grow_ch=opt.num_grow_ch, in_size=opt.val_crop)
        decoder = model.PriorDecoder(up_ch=opt.up_ch, sf=opt.sf, num_feat=opt.num_feat, num_out_ch=opt.num_out_ch)

    # Move network and containers to gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True if device == 'cuda' else False

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_root = 'run/trainx' + str(opt.sf) + '/saved_model/best_encoder.pth'
    decoder_root = 'run/trainx' + str(opt.sf) + '/saved_model/best_decoder.pth'
    encoder.load_state_dict(torch.load(encoder_root))
    decoder.load_state_dict(torch.load(decoder_root))

    # Initialize dataLoader
    valHRroot = 'datasets/{}/test/HR/'.format(opt.dataset)
    valLRroot = 'datasets/{}/test/lr_x{}'.format(opt.dataset, opt.sf)


    valDataset = MyDataset(valHRroot, valLRroot, opt.sf, opt.val_batch, 'test', opt.val_crop)
    valDataloader = DataLoader(valDataset, batch_size=opt.val_batch,
                                 shuffle=True, num_workers=opt.workers)

    # loss function
    MSE = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    MGE = GradLoss().to(device)

    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    running_mge = 0.0

    #evaluate
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for i, (lrim, hrim) in enumerate(valDataloader):
            lrim = lrim.to(device)
            hrim = hrim.to(device)

            if not opt.WithPriors:
                f1, f2 = encoder(lrim)
                sr_out = decoder(f1, f2)
            else:
                f1, f2, f3, f4 = encoder(lrim)
                sr_out = decoder(f1, f2, f3, f4)

            mse_loss = MSE(hrim, sr_out)
            ssim_value = ssim_loss(hrim * 0.5 + 0.5, sr_out * 0.5 + 0.5)
            mge_loss = MGE(hrim, sr_out)
            total_loss = mse_loss + (1 - ssim_value) + mge_loss
            topsnr_hr = torch.clamp(hrim* 0.5 + 0.5, min=0.0, max= 1.0)
            topsnr_sr = torch.clamp(sr_out * 0.5 + 0.5, min=0.0, max=1.0)
            psnr_value = psnr(topsnr_hr* 0.5 + 0.5, topsnr_sr* 0.5 + 0.5)

            running_loss += total_loss
            running_mse += mse_loss
            running_ssim += ssim_value
            running_psnr += psnr_value
            running_mge += mge_loss
            writer.add_figure('validation LR input vs. SR output vs. HR ',
                              plot_versus((((lrim[0] * 0.5) + 0.5) ),
                                          (((sr_out[0] * 0.5) + 0.5) ),
                                          (((hrim[0] * 0.5) + 0.5) )),
                              global_step=i)
            save_image(lrim * 0.5 + 0.5 , 'test_res/'+'x{}/{}'.format(opt.sf, opt.dataset) + 'lr'+str(i)+'.jpg')
            save_image(hrim * 0.5 + 0.5 , 'test_res/' + 'x{}/{}'.format(opt.sf, opt.dataset) + 'hr'+str(i)+'.jpg')
            save_image(torch.clamp(sr_out * 0.5+ 0.5,0.0,1.0) , 'test_res/' + 'x{}/{}'.format(opt.sf, opt.dataset) + 'sr'+str(i)+'.jpg')

        len_val = len(valDataset)/ opt.val_batch
        print(f"Test loss: {running_loss.item()/len_val:>7f}  Test MSE loss: {running_mse.item()/len_val:>7f} Test MGE loss: {running_mge.item()/len_val:>7f}")
        print(f"Test SSIM: {running_ssim.item()/len_val:>7f}  Test PSNR: {running_psnr.item()/len_val:>7f}")
        writer.close()
        print('Test Finished')

if __name__=="__main__":


    # The detail network setting
    opt = parser.parse_args()
    opt.sf =3
    opt.base_interpolate = 1
    #'Set5', 'Set14',
    for data in ['Set5', 'Set14','BSD100', 'Urban100']:
        opt.dataset = data
        print(opt)
        main(opt)
    sys.exit()



