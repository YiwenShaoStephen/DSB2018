import argparse
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as t
from torchvision import transforms as tsf
from pathlib import Path
from skimage import io
import PIL
from models.Unet_naive import UNet
from skimage.transform import resize
from skimage.morphology import label

TRAIN_PATH = './train.pth.tar'
TEST_PATH = './test.pth.tar'

parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    help='print frequency (default: 5)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--depth', default=5, type=int,
                    help='Number of conv blocks.')
parser.add_argument('--name', default='Unet-5', type=str,
                    help='name of experiment')


def main():
    global args, best_prec1
    args = parser.parse_args()
    if not os.path.exists(TEST_PATH):
        test = process('./input/stage1_test/', False)
        t.save(test, TEST_PATH)
    if not os.path.exists(TRAIN_PATH):
        train_data = process('./input/stage1_train/')
        t.save(train_data, TRAIN_PATH)

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((128, 128)),
        tsf.ToTensor(),
        tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    t_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((128, 128), interpolation=PIL.Image.NEAREST),
        tsf.ToTensor(),
    ])

    trainset = TrainDataset(TRAIN_PATH, s_trans, t_trans)
    trainloader = t.utils.data.DataLoader(
        trainset, num_workers=2, batch_size=32)

    testset = TestDataset(TEST_PATH, s_trans)
    testloader = t.utils.data.DataLoader(testset, num_workers=1, batch_size=1)

    # Train
    #model = UNet(1, in_channels=3, depth=args.depth).cuda()
    model = UNet(3, 1).cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):

        losses = AverageMeter()

        for i, (x_train, y_train) in enumerate(trainloader):
            x_train = t.autograd.Variable(x_train).cuda()
            y_train = t.autograd.Variable(y_train).cuda()
            optimizer.zero_grad()
            o = model(x_train)
            loss = soft_dice_loss(o, y_train)

            losses.update(loss.data[0], args.batch_size)

            loss.backward()
            optimizer.step()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          epoch, i, len(trainloader), loss=losses))

    # Test
    model = model.eval()
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


# Preprocess data and save it to disk
def process(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        imgs = []
        for image in (file / 'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs) == 1
        if img.shape[2] > 3:
            assert(img[:, :, 3] != 255).sum() == 0
        img = img[:, :, :3]

        if has_mask:
            mask_files = list((file / 'masks').iterdir())
            masks = None
            for ii, mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask != 0)] == 255).all()
                if masks is None:
                    H, W = mask.shape
                    masks = np.zeros((len(mask_files), H, W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask != 0] == 255).all()
            for ii, mask in enumerate(masks):
                masks[ii] = mask / 255 * (ii + 1)
            mask = masks.sum(0)
            item['mask'] = t.from_numpy(mask)
        item['name'] = str(file).split('/')[-1]
        item['img'] = t.from_numpy(img)
        datas.append(item)
    return datas


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
        mask = data['mask'][:, :, None].byte().numpy()

        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.datas)


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
