from unet.unet_model3 import *


unet = UNet(3, 1)
unet.eval()
unet.load_state_dict(th.load('.\checkpoint\\pretrain\\PersonMasker_model32972.pt'))
params = unet.state_dict()
print()