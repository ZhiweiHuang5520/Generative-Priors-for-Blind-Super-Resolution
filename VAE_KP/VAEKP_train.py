import torch.optim as optim
import time
import network
from batchLoader import *
from pathlib import Path
import sys


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        results = model.forward(data, labels = label)
        loss = model.loss_function(*results,
                                              M_N = 0.00025, #al_img.shape[0]/ self.num_train_imgs,
                                              )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss

def test(model, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            data = data.to(device)
            results = model.forward(data)
            loss = model.loss_function(*results,
                                       M_N=0.00025,  # al_img.shape[0]/ self.num_train_imgs,
                                       )

            test_loss += loss.item()

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    return test_loss

def main():
    # parameters
    BATCH_SIZE = 512
    TEST_BATCH_SIZE = 256
    EPOCHS = 200

    # args.hidden_size = min(args.sf * 5 + 5, 25)
    LATENT_DIMENSION = 15
    LEARNING_RATE = 1e-3

    # default=11,help='Kernel size. 11, 15, 19 for x2, x3, x4; to be overwritten automatically') args.kernel_size = min(args.sf * 4 + 3, 21)
    kernel_size = 11
    scale_factor = 2

    USE_CUDA = True
    PRINT_INTERVAL = 100
    MODEL_PATH = './x{}/'.format(scale_factor)
    val_save_path = '../data/datasets/Kernel_validation_set'
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)




    USE_CUDA = True

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)


    # training code


    data_train = KernelFolder(val_save_path, train=True, kernel_size=kernel_size, scale_factor=scale_factor)
    data_test = KernelFolder(val_save_path, train=False, kernel_size=kernel_size, scale_factor=scale_factor)

    print('num train_images:', len(data_train))
    print('num test_images:', len(data_test))


    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    print('latent size:', LATENT_DIMENSION)

    model = network.BetaVAE(in_channels= 1, kernel_size = kernel_size, latent_dim = LATENT_DIMENSION ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,150], gamma=0.1)
    save_loss = float('inf')
    for epoch in range(EPOCHS):
        train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_loss = test(model, device, test_loader)
        scheduler.step()
        if test_loss<save_loss:
            torch.save(model.state_dict(), MODEL_PATH + 'best.pth' )
            save_loss = test_loss
if __name__ == "__main__":
    main()
    sys.exit()

