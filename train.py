import torch as th
from Loss import *
from unet.unet_model import *
from utils.load_dataset import *
import torch.optim as optim
from SummaryWriter import SummaryWriter
from utils.monitor import *

batch_size = 64*6
epochs = 100000


PersonTrainImage = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\image_64'
PersonTrainMask = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\mask_64'
PersonBbox = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\bbox_64_128'


unet = UNet(3, 1).to('cuda')
unet.eval()
conf = confconv(64).to('cuda')
conf.train()
writer = SummaryWriter('.\log\log.mat')
unet.load_state_dict(th.load('E:\Person_detection\Mask_Yolo\checkpoint\\pretrain\\PersonMasker100.pt'))

dataSet = load_dataset(PersonTrainImage, PersonTrainMask, PersonBbox)
trainSet, valSet = split_train_val(dataSet, val_percent=0.2)


trainLoader = DataLoader(trainSet, batch_size)
valLoader = DataLoader(trainSet, batch_size)


optimizer = optim.Adadelta(conf.parameters(), lr=1e-4)
max_acc = 1e-8
for i in range(epochs):
    sum_loss = 0
    for j in range(trainLoader.num_step):
        image, mask, bbox = trainLoader.next_batch_cat(8, 512)
        pre_mask, pre_box, _, pre_conf = unet(th.cuda.FloatTensor(image))
        pre_conf = conf(pre_conf)
        loss_mask, loss_box, loss_conf = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask),
                                                   pre_box=pre_box, target_box=th.cuda.FloatTensor(bbox),
                                                   pre_conf=pre_conf)
        loss = loss_conf
        loss.backward()
        optimizer.step()
        ###############################################################
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)
        mIOU, IOU = mIou(pre_box=pre_box.detach().cpu().numpy(), target_box=bbox)
        acc_conf, recall_conf, fscore_conf = confMonitor(IOU, pre_conf.detach().cpu().numpy(), 0.5)

        print('train epoch', i, 'step', j, 'loss', float(loss), 'max_acc,', max_acc, 'loss_mask',
               float(loss_mask), 'loss_box', float(loss_box))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero, 'mIOU', mIOU,
              'acc_conf', acc_conf, 'recall_conf', recall_conf)

        writer.write('trainloss', float(loss))
        writer.write('train_acc_one', acc_one)
        writer.write('train_recall_one', recall_one)
        writer.write('train_acc_zero', acc_zero)
        writer.write('train_recall_zero', recall_zero)
        writer.write('mIOU', mIOU)
        writer.write('acc_conf', acc_conf)
        writer.write('recall_conf', recall_conf)
    for k in range(valLoader.num_step):
        image, mask, bbox = valLoader.next_batch_cat(8, 512)
        pre_mask, pre_box, _, pre_conf = unet(th.cuda.FloatTensor(image))
        pre_conf = conf(pre_conf)
        loss_mask, loss_box, loss_conf = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask),
                                                   pre_box=pre_box, target_box=th.cuda.FloatTensor(bbox),
                                                   pre_conf=pre_conf)
        mIOU, IOU = mIou(pre_box=pre_box.detach().cpu().numpy(), target_box=bbox)
        acc_conf, recall_conf, fscore_conf = confMonitor(IOU, pre_conf.detach().cpu().numpy(), 0.5)
        loss = loss_conf
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)

        sum_loss += float(acc_conf)
        print('val epoch', i, 'step', k, 'loss', float(loss), 'max_acc,', max_acc, 'loss_mask',
              float(loss_mask), 'loss_box', float(loss_box))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero)
        # print('recall_one2', recall_one2, 'acc_one2', acc_one2, 'recall_zero2', recall_zero2, 'acc_zero2', acc_zero2)
        writer.write('valloss', float(loss))
        writer.write('val_acc_one', acc_one)
        writer.write('val_recall_one', recall_one)
        writer.write('val_acc_zero', acc_zero)
        writer.write('val_recall_zero', recall_zero)

    if sum_loss / valLoader.num_step > max_acc:
        print('*******************************')
        print('max_acc=', max_acc)
        th.save(unet.conf.state_dict(), 'checkpoint\conf{}.pt'.format(str(i)))
        max_acc = sum_loss / valLoader.num_step
        sum_loss = 0
    writer.savetomat()



