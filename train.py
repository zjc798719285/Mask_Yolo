from Loss import *
from unet.unet_model4 import *
from utils.load_dataset_h5 import *
import torch.optim as optim
from SummaryWriter import SummaryWriter
from utils.monitor import *
batch_size256 = 4 * 6
batch_size128 = 16 * 6
batch_size64 = 64 * 6
epochs = 100000


tensor = th.zeros(6, 10, 128, 128)
unet = UNet(3, 1, tensor).to('cuda')
unet.train()
writer = SummaryWriter('.\log\log.mat')


# trainLoader64 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\train_64.h5', batch_size64)
# valLoader64 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\val_64.h5', batch_size64)
trainLoader128 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\sub_train_128.h5', batch_size128)
valLoader128 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\sub_val_128.h5', batch_size128)
# trainLoader256 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\train_256.h5', batch_size256)
# valLoader256 = DataLoader('E:\Person_detection\Dataset\DataSets2017\\u_net\\val_256.h5', batch_size256)


optimizer = optim.Adadelta(unet.parameters(), lr=1e-4)
max_acc = 1e-8
train_switch = -1
val_switch = -1
for i in range(epochs):
    sum_loss = 0
    for j in range(trainLoader128.num_step):

        image, mask, bbox, mask_res = trainLoader128.next_batch_cat(4, 512, 4)
        pre_mask, pre_box, pre_mask_res = unet(th.cuda.FloatTensor(image))
        loss_mask, loss_box = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask),
                                        pre_box=pre_box, target_box=th.cuda.FloatTensor(bbox),
                                        pre_mask_res=pre_mask_res, target_mask_res=th.cuda.FloatTensor(mask_res))
        loss = loss_mask + loss_box
        loss.backward()
        optimizer.step()
        ###############################################################
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)
        mIOU, IOU = mIou(pre_box=pre_box.detach().cpu().numpy(), target_box=bbox)
        print('train epoch', i, 'step', j, 'loss', float(loss), 'max_acc,', max_acc, 'loss_mask',
               float(loss_mask), 'loss_box', float(loss_box))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero,
              'mIOU', mIOU)

        writer.write('trainloss', float(loss))
        writer.write('train_acc_one', acc_one)
        writer.write('train_recall_one', recall_one)
        writer.write('train_acc_zero', acc_zero)
        writer.write('train_recall_zero', recall_zero)
        writer.write('loss_box', float(loss_box))
        writer.write('mIOU', mIOU)
        writer.write('loss_mask', float(loss_mask))


    for k in range(valLoader128.num_step):

        image, mask, bbox, mask_res = valLoader128.next_batch_cat(4, 512, 4)

        pre_mask, pre_box, pre_mask_res = unet(th.cuda.FloatTensor(image))

        loss_mask, loss_box = unet_loss(pre_mask=pre_mask, target_mask=th.cuda.FloatTensor(mask),
                                        pre_box=pre_box, target_box=th.cuda.FloatTensor(bbox),
                                        pre_mask_res=pre_mask_res, target_mask_res=th.cuda.FloatTensor(mask_res))

        mIOU, IOU = mIou(pre_box=pre_box.detach().cpu().numpy(), target_box=bbox)
        loss = loss_mask + loss_box
        recall_one, acc_one, recall_zero, acc_zero = recall_ap(pre=pre_mask.detach().cpu().numpy(), target=mask, cls=0)

        sum_loss += float(0.5*(recall_one + recall_zero) + mIOU)
        print('val epoch', i, 'step', k, 'loss', float(loss), 'max_acc,', max_acc, 'loss_mask',
              float(loss_mask), 'loss_box', float(loss_box))
        print('recall_one', recall_one, 'acc_one', acc_one, 'recall_zero', recall_zero, 'acc_zero', acc_zero)

        writer.write('valloss', float(loss))
        writer.write('val_acc_one', acc_one)
        writer.write('val_recall_one', recall_one)
        writer.write('val_acc_zero', acc_zero)
        writer.write('val_recall_zero', recall_zero)
    writer.write('current_acc', sum_loss / valLoader128.num_step)
    if sum_loss / valLoader128.num_step > max_acc:
        print('*******************************')
        print('max_acc=', max_acc)
        th.save(unet.state_dict(), 'checkpoint\PersonMaskerUnitBox_{}.pt'.format(str(i)))
        max_acc = sum_loss / valLoader128.num_step
        sum_loss = 0

    if i % 10 == 0:
        print('*******************************')
        print('max_acc=', max_acc)
        th.save(unet.state_dict(), 'checkpoint\PersonMaskerUnitBox10_{}.pt'.format(str(i)))
    writer.savetomat()



