import torch
from model import RecNet
from train import modelTrain
from data import DataFeed
import torchvision.transforms as trf
from torch.utils.data import DataLoader
import scipy.io as sio
import os.path as path
import matplotlib.pyplot as plt
from skimage import io, transform

import pdb # For debug model

# Experiment options:
data_dir = path.join(path.dirname(path.dirname(__file__)), 'data', 'dev_dataset_csv')
options_dict = {
    'tag': 'Exp1_beam_seq_pred_no_images',
    'operation_mode': 'beams',

    # Data:
    'train_ratio': 1,
    'test_ratio': 1,
    'img_mean': (0.4905,0.4938,0.5285),
    'img_std':(0.05922,0.06468,0.06174),
    'trn_data_file': '../'+data_dir+'/train_set.csv',
    'val_data_file': '../'+data_dir+'/val_set.csv',
    'results_file': '../checkpoints/whitenoise+def_img_5.mat',
    'model_file': '../checkpoints/whitenoise+def_img_5.pt',

    # Noise
    'noise_dict' : {
        'dev_def_image_prob': 0.002,
        'dev_def_beams_prob': 0,
        'wh_noi_std_beam': 0.5,
        'wh_noi_std_image': 2, # 255
        'brightness_change_var': 0, #1
        'brightness_change_prob': 0,
        'hue_std': 0, #180
        'hue_change_prob': 0
    },



    # Net:
    'net_type':'gru',
    'cb_size': 128,  # Beam codebook size
    'out_seq': 5,  # Length of the predicted sequence
    'inp_seq': 8, # Length of inp beam and image sequence
    'embed_dim': 50,  # Dimension of the embedding space (for beam indices)
    'embed_dim_img': 256,  # Dimension of the embedding space (for images)
    'hid_dim': 128,  # Dimension of the hidden state of the RNN
    'attention_dim': 128,
    'img_dim': [3, 160, 256],  # Dimensions of the input image
    'out_dim': 128,  # Dimensions of the softmax layers
    'num_rec_lay': 2,  # Depth of the recurrent network
    'drop_prob': 0.3,

    # Train param
    'gpu_idx': 0,
    'solver': 'Adam',
    'shf_per_epoch': True,
    'num_epochs': 15,
    'batch_size': 32,
    'val_batch_size': 32,
    'lr': 1e-3,
    'lr_sch': [200],
    'lr_drop_factor':0.1,
    'wd': 0,
    'display_freq': 1000,
    'coll_cycle': 200,
    'val_freq': 2000,
    'prog_plot': True,
    'fig_c': 0,
    'SIGMA': 0.5,
    "fine_tune_encoder" : False,  # fine-tune encoder?
    'patience': 3   # Early stop
}
fine_tune_encoder = options_dict['fine_tune_encoder']
encoder_lr = 1e-4

# Fetch training data

# resize = trf.Resize((options_dict['img_dim'][1],options_dict['img_dim'][2]))
normalize = trf.Normalize(mean=options_dict['img_mean'],
                          std=options_dict['img_std'])
# transf = trf.Compose([
#     trf.ToPILImage(),
#     resize,
#     trf.ToTensor(),
#     normalize
# ])
trn_feed = DataFeed(root_dir=options_dict['trn_data_file'],
                     n=options_dict['inp_seq']+options_dict['out_seq'],
                     img_dim=tuple(options_dict['img_dim']),
                     transform=normalize,
                     noise_dict = options_dict['noise_dict'])
trn_loader = DataLoader(trn_feed, batch_size=options_dict['batch_size'])
options_dict['train_size'] = trn_feed.__len__()

val_feed = DataFeed(root_dir=options_dict['val_data_file'],
                     n=options_dict['inp_seq']+options_dict['out_seq'],
                     img_dim=tuple(options_dict['img_dim']),
                     transform=normalize, 
                     noise_dict = options_dict['noise_dict'])
val_loader = DataLoader(val_feed,batch_size=options_dict['val_batch_size'])
options_dict['test_size'] = val_feed.__len__()

with torch.cuda.device(options_dict['gpu_idx']):

    # Build net:
    # ----------
    if options_dict['net_type'] == 'gru':
        net = RecNet(options_dict['embed_dim'],
                     options_dict['cb_size'],
                     options_dict['embed_dim_img'],
                     options_dict['hid_dim'],
                     options_dict['inp_seq'],
                     options_dict['attention_dim'],
                     options_dict['out_dim'],
                     options_dict['out_seq'],
                     options_dict['num_rec_lay'],
                     options_dict['drop_prob'],
                     )
        net = net.cuda()
        # encoder = Encoder(options_dict['embed_dim_img'])
        # encoder = encoder.cuda()

    # Train and test:
    # ---------------
    net, options_dict, train_info = modelTrain(net,
                                               trn_loader,
                                               val_loader,
                                               options_dict)

    # Plot progress:
    if options_dict['prog_plot']:
        # options_dict['fig_c'] += 1
        fig = plt.figure()
        plt.plot(train_info['train_itr'], train_info['train_top_1'],'-or', label='Train top-1')
        # plt.plot(train_info['train_itr'], train_info['train_top_1_score'],'-or', label='Train top-1')
        plt.plot(train_info['val_itr'], train_info['val_top_1'],'-ob', label='Validation top-1')
        # plt.plot(train_info['val_itr'], train_info['val_top_1_score'],'-ob', label='Validation top-1')
        plt.xlabel('Training iteration')
        plt.ylabel('Top-1 accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('whitenoise_def_img_5.jpg')
        plt.close(fig)

    sio.savemat(options_dict['results_file'],train_info)
