import torch as th
from Loss import *
from unet.unet_model3 import *
from utils.load_dataset2 import *
import torch.optim as optim
from SummaryWriter import SummaryWriter
from utils.monitor import *
import time
batch_size128 = 16 * 6
batch_size64 = 64 * 6
epochs = 100000

PersonTrainImage128 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\image_128'
PersonTrainMask128 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\mask_128'
PersonBbox128 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\bbox_128'

#
PersonTrainImage64 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\image_64'
PersonTrainMask64 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\mask_64'
PersonBbox64 = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\bbox_64'



unet = UNet(3, 1).to('cuda')
unet.train()
writer = SummaryWriter('.\log\log.mat')
unet.load_state_dict(th.load('E:\Person_detection\Mask_Yolo\checkpoint\\pretrain\\PersonMasker_model3140.pt'))

dataSet128 = load_dataset(PersonTrainImage128, PersonTrainMask128, PersonBbox128)
trainSet128, valSet128 = split_train_val(dataSet128, val_percent=0.2)
dataSet64 = load_dataset(PersonTrainImage64, PersonTrainMask64, PersonBbox64, )
trainSet64, valSet64 = split_train_val(dataSet64, val_percent=0.2)

#
trainLoader128 = DataLoader(trainSet128, batch_size128)
valLoader128 = DataLoader(valSet128, batch_size128)
trainLoader64 = DataLoader(trainSet64, batch_size64)
valLoader64 = DataLoader(valSet64, batch_size64)

optimizer = optim.Adadelta(unet.conf.parameters(), lr=1e-4)
max_acc = 1e-8
for i in range(epochs):
    sum_loss = 0
    for j in range(trainLoader128.num_step):
        if j % 5 == 1e10:
            image, mask, bbox = trainLoader64.next_batch_cat(8, 512, 4)
        else:
            image, mask, bbox = trainLoader128.next_batch_cat(4, 512, 4)

        pre_mask, pre_box, pre_conf, _ = unet(th.cuda.FloatTensor(image))

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
    for k in range(valLoader128.num_step):
        if k % 5 == 1e10:
            image, mask, bbox = trainLoader64.next_batch_cat(8, 512, 4)
        else:
            image, mask, bbox = trainLoader128.next_batch_cat(4, 512, 4)
        pre_mask, pre_box, pre_conf, _ = unet(th.cuda.FloatTensor(image))

        loss_mask, loss_box, loss_conf = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask),
                                                   pre_box=pre_box, target_box=th.cuda.FloatTensor(bbox),
                                                   pre_conf=pre_conf)
        mIOU, IOU = mIou(pre_box=pre_box.detach().cpu().numpy(), target_box=bbox)
        acc_conf, recall_conf, fscore_conf = confMonitor(IOU, pre_conf.detach().cpu().numpy(), 0.5)
        loss = loss_conf
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)

        sum_loss += float(fscore_conf)
        print('val epoch', i, 'step', k, 'loss', float(loss), 'max_acc,', max_acc, 'loss_mask',
              float(loss_mask), 'loss_box', float(loss_box))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero)
        # print('recall_one2', recall_one2, 'acc_one2', acc_one2, 'recall_zero2', recall_zero2, 'acc_zero2', acc_zero2)
        writer.write('valloss', float(loss))
        writer.write('val_acc_one', acc_one)
        writer.write('val_recall_one', recall_one)
        writer.write('val_acc_zero', acc_zero)
        writer.write('val_recall_zero', recall_zero)

    if sum_loss / valLoader128.num_step > max_acc:
        print('*******************************')
        print('max_acc=', max_acc)
        th.save(unet.state_dict(), 'checkpoint\PersonMasker_model3{}.pt'.format(str(i)))
        max_acc = sum_loss / valLoader128.num_step
        sum_loss = 0
    writer.savetomat()



