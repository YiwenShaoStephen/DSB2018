import sys
import argparse
import os
import shutil
import time
import torchvision
import random
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as t
from torchvision import transforms as tsf
from skimage.io import imread
from models.Unet import UNet
from skimage.transform import resize
from skimage.morphology import label
from tensorboard_logger import configure, log_value


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--depth', default=5, type=int,
                    help='Number of conv blocks.')
parser.add_argument('--img-height', default=128, type=int,
                    help='Height of resized images')
parser.add_argument('--img-width', default=128, type=int,
                    help='width of resized images')
parser.add_argument('--img-channels', default=3, type=int,
                    help='Number of channels of images')
parser.add_argument('--name', default='Unet-5', type=str,
                    help='name of experiment')
parser.add_argument('--train-path', default='./input/stage1_train/', type=str,
                    help='Path of raw training data')
parser.add_argument('--test-path', default='./input/stage1_test/', type=str,
                    help='Path of raw test data')
parser.add_argument('--train-data', default='./train.pth.tar', type=str,
                    help='Path of processed training data')
parser.add_argument('--val-data', default='./val.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--test-data', default='./test.pth.tar', type=str,
                    help='Path of processed test data')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_false')


best_loss = 1
random.seed(0)


def main():
    global args, best_loss
    args = parser.parse_args()

    if args.tensorboard:
        print("Using tensorboard")
        configure("exp/%s" % (args.name))

    if not (os.path.exists(args.train_data) and
            os.path.exists(args.train_data) and
            os.path.exists(args.test_data)):
        train, val, test = DataProcess(args.train_path, args.test_path, 0.9,
                                       args.img_channels)
        t.save(train, args.train_data)
        t.save(val, args.val_data)
        t.save(test, args.test_data)

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((args.img_height, args.img_width)),
        tsf.ToTensor(),
    ])

    t_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((args.img_height, args.img_width),
                   interpolation=PIL.Image.NEAREST),
        tsf.ToTensor(),
    ])

    # split the training set into training set and validation set
    trainset = TrainDataset(args.train_data, s_trans, t_trans)
    trainloader = t.utils.data.DataLoader(
        trainset, num_workers=1, batch_size=args.batch_size, shuffle=True)

    valset = TrainDataset(args.val_data, s_trans, t_trans)
    valloader = t.utils.data.DataLoader(
        valset, num_workers=1, batch_size=args.batch_size)

    NUM_TRAIN = len(trainset)
    NUM_VAL = len(valset)
    NUM_ALL = NUM_TRAIN + NUM_VAL
    print('Total samples: {0} \n'
          'Using {1} samples for training, '
          '{2} samples for validation'.format(NUM_ALL, NUM_TRAIN, NUM_VAL))

    testset = TestDataset(args.test_data, s_trans)
    testloader = t.utils.data.DataLoader(testset, num_workers=1, batch_size=1)

    # create model
    model = UNet(1, in_channels=3, depth=args.depth).cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = t.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(args.start_epoch, args.epochs):
        Train(trainloader, model, optimizer, epoch)
        val_loss = Validate(valloader, model, epoch)
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_loss,
        }, is_best)
    print 'Best validation loss: ', best_loss

    # Visualize some predicted masks on training data to get a better intuition
    # about the performance. Comment it if not necessary.
    datailer = iter(trainloader)
    img, mask = datailer.next()
    torchvision.utils.save_image(img, 'raw.png')
    torchvision.utils.save_image(mask, 'mask.png')
    img = t.autograd.Variable(img).cuda()
    img_pred = model(img)
    img_pred = img_pred.data
    torchvision.utils.save_image(img_pred > 0.5, 'predicted.png')

    # Load the best model and evaluate on test set
    checkpoint = t.load('exp/%s/' %
                        (args.name) + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    Test(testloader, model)


def DataProcess(TRAIN_PATH, TEST_PATH, train_prop, channels):
    # Get train and test IDs
    train_all_ids = next(os.walk(TRAIN_PATH))[1]
    # split the training set into train and validation set
    random.shuffle(train_all_ids)

    num_all = len(train_all_ids)
    num_train = int(num_all * train_prop)
    train_ids = train_all_ids[:num_train]
    val_ids = train_all_ids[num_train:]

    test_ids = next(os.walk(TEST_PATH))[1]

    train = []
    val = []
    test = []

    print('Getting train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        train_item = {}
        train_item['name'] = id_
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :channels]
        # img = resize(img, (height, width),
        #              mode='constant', preserve_range=True)

        train_item['img'] = t.from_numpy(img)
        mask = None
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            if mask is None:
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
        train_item['mask'] = t.from_numpy(mask)
        train.append(train_item)

    print('Getting validation images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
        val_item = {}
        val_item['name'] = id_
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :channels]
        val_item['img'] = t.from_numpy(img)
        mask = None
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            if mask is None:
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
        val_item['mask'] = t.from_numpy(mask)
        val.append(val_item)

    # Get and resize test images
    print('Getting test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        test_item = {}
        test_item['name'] = id_
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :channels]
        test_item['img'] = t.from_numpy(img)
        test.append(test_item)

    print('Done!')
    return train, val, test


# Loss definition
# Use Soft Dice Loss
def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def Train(trainloader, model, optimizer, epoch):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for i, (x_train, y_train) in enumerate(trainloader):
        x_train = t.autograd.Variable(x_train.cuda())
        y_train = t.autograd.Variable(y_train.cuda(async=True))

        optimizer.zero_grad()
        o = model(x_train)
        loss = soft_dice_loss(o, y_train)

        losses.update(loss.data[0], args.batch_size)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      loss=losses))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)


def Validate(validateloader, model, epoch):
    """Perform validation on the validation set"""
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (x_val, mask) in enumerate(validateloader):
        mask = mask.cuda(async=True)
        x_val = x_val.cuda()
        x_val = t.autograd.Variable(x_val, volatile=True)
        mask = t.autograd.Variable(mask, volatile=True)
        output = model(x_val)
        loss = soft_dice_loss(output, mask)

        losses.update(loss.data[0], args.batch_size)

        if i % args.print_freq == 0:
            print('Val: [{0}][{1}/{2}]\t'
                  'Val Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(validateloader), loss=losses))

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)

    return losses.avg


def Test(testloader, model):
    """Evaluate on the test set and save run-length encoding results to disk"""
    model.eval()
    predictions = []
    test_ids = []

    for input, size, id in testloader:
        input_var = t.autograd.Variable(input, volatile=True).cuda()
        pred_test = model(input_var)
        pred_test = pred_test[0][0].data.cpu().numpy()
        pred_up = resize(pred_test, (size[0].numpy()[0], size[1].numpy()[0]),
                         preserve_range=True, mode='reflect')
        predictions.append(pred_up)
        test_ids.append(id[0])

        new_test_ids = []
        rles = []
        for n, id_ in enumerate(test_ids):
            rle = list(prob_to_rles(predictions[n], 0.5))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(
        lambda x: ' '.join(str(y) for y in x))
    directory = "exp/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    csvname = directory + "sub-dsbowl2018.csv"
    sub.to_csv(csvname, index=False)


# run length encoding
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.1):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding((lab_img == i).astype(int))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "exp/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    t.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'exp/%s/' %
                        (args.name) + 'model_best.pth.tar')


# Pytorch customized dataload
class TrainDataset():
    def __init__(self, path, source_transform, target_transform):
        self.datas = t.load(path)
        self.s_transform = source_transform
        self.t_transform = target_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:, :, None].numpy()

        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.datas)


# class ChunkSampler(sampler.Sampler):
#     """Samples elements sequentially from some offset.
#     Arguments:
#         num_samples: # of desired datapoints
#         start: offset where we should start selecting from
#     """

#     def __init__(self, num_samples, start=0):
#         self.num_samples = num_samples
#         self.start = start

#     def __iter__(self):
#         return iter(range(self.start, self.start + self.num_samples))

#     def __len__(self):
#         return self.num_samples


class TestDataset():
    def __init__(self, path, source_transform):
        self.datas = t.load(path)
        self.s_transform = source_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        size = img.shape
        id = data['name']
        img = self.s_transform(img)
        return img, size, id

    def __len__(self):
        return len(self.datas)


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


if __name__ == '__main__':
    main()
