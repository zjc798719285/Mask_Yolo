import torch as th
from Loss import *
from unet.unet_model3 import UNet
from utils.load_dataset import *
import torch.optim as optim
from SummaryWriter import SummaryWriter
from utils.monitor import *

batch_size = 64*6
epochs = 100000


PersonTrainImage = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\image_64'
PersonTrainMask = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\mask_64'


unet = UNet(3, 3).to('cuda')
unet.train()
writer = SummaryWriter('.\log\log.mat')
# unet.load_state_dict(th.load('E:\Person_detection\Pytorch-UNet\checkpoint\\PersonMasker52.pt'))

dataSet = load_dataset(PersonTrainImage, PersonTrainMask)
trainSet, valSet = split_train_val(dataSet, val_percent=0.1)


trainLoader = DataLoader(trainSet, batch_size)
valLoader = DataLoader(valSet, batch_size)


optimizer = optim.Adadelta(unet.parameters(), lr=1e-4)
max_acc = 0
for i in range(epochs):
    sum_loss = 0
    for j in range(trainLoader.num_step):
        image, mask = trainLoader.next_batch_cat(8, 512)
        pre_mask1, pre_mask2 = unet(th.cuda.FloatTensor(image))
        loss1 = unet_loss(pre_mask=pre_mask1, target_mask=th.cuda.FloatTensor(mask))
        loss2 = unet_loss(pre_mask=pre_mask2, target_mask=th.cuda.FloatTensor(mask))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        #########################
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask1.detach().cpu().numpy(), target=mask, cls=0)
        recall_one2, acc_one2, recall_zero2, acc_zero2 = recall_ap(pre=pre_mask2.detach().cpu().numpy(), target=mask, cls=0)
        # recall_car, acc_car = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=1)
        # recall_back, acc_back = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=2)
        print('train epoch', i, 'step', j, 'loss', float(loss1), 'max_acc,', max_acc,
              'per_loss', float(loss1))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero)
        print('recall_one2', recall_one2, 'acc_one2', acc_one2, 'recall_zero2', recall_zero2, 'acc_zero2', acc_zero2)
        writer.write('trainloss', float(loss))
        writer.write('train_acc_one', acc_one)
        writer.write('train_recall_one', recall_one)
        writer.write('train_acc_zero', acc_zero)
        writer.write('train_recall_zero', recall_zero)

        # writer.write('trainCarloss', float(loss2))
        # writer.write('trainBackloss', float(loss3))

    for k in range(valLoader.num_step):
        image, mask = valLoader.next_batch_cat(8, 512)
        pre_mask, pre_mask2 = unet(th.cuda.FloatTensor(image))
        loss1 = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask))
        loss2 = unet_loss(pre_mask=pre_mask2, target_mask=th.cuda.FloatTensor(mask))
        loss = loss1 + loss2
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)
        recall_one2, acc_one2, recall_zero2, acc_zero2 = recall_ap(pre=pre_mask2.detach().cpu().numpy(), target=mask, cls=0)
        sum_loss += float(acc_one2)
        print('val epoch', i, 'step', k, 'loss', float(loss), 'max_acc,', max_acc,
              'per_loss', float(loss1))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero)
        print('recall_one2', recall_one2, 'acc_one2', acc_one2, 'recall_zero2', recall_zero2, 'acc_zero2', acc_zero2)
        writer.write('valloss', float(loss))
        writer.write('val_acc_one', acc_one2)
        writer.write('val_recall_one', recall_one2)
        writer.write('val_acc_zero', acc_zero2)
        writer.write('val_recall_zero', recall_zero2)

    if sum_loss / valLoader.num_step > max_acc:
        print('*******************************')
        print('max_acc=', max_acc)
        th.save(unet.state_dict(), 'checkpoint\PersonMasker{}.pt'.format(str(i)))
        max_acc = sum_loss / valLoader.num_step
        sum_loss = 0
    writer.savetomat()



