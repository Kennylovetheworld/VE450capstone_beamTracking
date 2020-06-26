import torch
import torch.nn as nn
import torch.optim as optimizer
# from torch.utils.data import DataLoader
import numpy as np
import time
import pdb
from tqdm import tqdm

alpha_c = 1.

class Callback():
    def __init__(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self): pass
    def on_validation_begin(self): pass
    def on_validation_end(self): pass

class Training(Callback):
    def __init__(self, embed, options_dict):
        self.embed = embed
        self.options_dict = options_dict

    def on_batch_begin(self, beams, images, train=True):
        init_beams = beams[:, :self.options_dict['inp_seq']].type(torch.LongTensor)
        inp_beams = self.embed(init_beams)
        inp_beams = inp_beams.cuda()
        targ = beams[:, self.options_dict['inp_seq']:self.options_dict['inp_seq']+self.options_dict['out_seq']]\
                .type(torch.LongTensor)
        if train:
            targ = targ.view(-1)
        targ = targ.cuda()
        batch_size = beams.shape[0]
        return inp_beams, targ, batch_size

class Validation(Callback):
    def __init__(self, options_dict):
        self.running_val_top_1 = []
        self.running_val_top_1_score = []
        self.val_acc_ind = []
        self.options_dict = options_dict
        self.best_score = 0.0
        self.count = 0

    def on_validation_end(self, model, itr, batch_acc, batch_score):
        self.running_val_top_1.append(batch_acc.cpu().numpy() / self.options_dict['test_size'])
        self.running_val_top_1_score.append(batch_score.cpu().numpy() / self.options_dict['test_size'])
        self.val_acc_ind.append(itr)
        print('Validation-- Top-1 accuracy = {0:5.4f} and Top-1 score = {1:5.4f}'.format(
                self.running_val_top_1[-1],
                self.running_val_top_1_score[-1]
            )
        )
        # Early stop
        if self.running_val_top_1_score[-1] > self.best_score:
            self.best_score = self.running_val_top_1_score[-1]
            self.count = 0
            # Store model
            torch.save(model.state_dict(), self.options_dict['model_file'])
        else:
            self.count += 1
        
        return self.count >= self.options_dict['patience']
    
    def get_stat(self):
        return self.running_val_top_1, self.running_val_top_1_score, self.val_acc_ind


def modelTrain(net,trn_loader,val_loader,options_dict):
    """
    :param net:
    :param data_samples:
    :param options_dict:
    :return:
    """
    
    
    # Optimizer:
    # ----------
    if options_dict['solver'] == 'Adam':
        opt = optimizer.Adam(net.parameters(),
                             lr=options_dict['lr'],
                             weight_decay=options_dict['wd'],
                             amsgrad=True)
    else:
        ValueError('Not recognized solver')

    scheduler = optimizer.lr_scheduler.MultiStepLR(opt,
                                                   milestones=options_dict['lr_sch'],
                                                   gamma=options_dict['lr_drop_factor'])

    # Define training loss:
    # ---------------------
    criterion = nn.CrossEntropyLoss()

    # Initialize training hyper-parameters:
    # -------------------------------------
    itr = 0
    embed = nn.Embedding(options_dict['cb_size'], options_dict['embed_dim'])
    running_train_loss = []
    running_trn_top_1 = []
    running_trn_top_1_score = []
    train_loss_ind = []
    stop = False
    # Initialize Callbacks
    # ------------------------
    traincb = Training(embed, options_dict)
    valcb = Validation(options_dict)


    print('------------------------------- Commence Training ---------------------------------')
    t_start = time.clock()
    for epoch in range(options_dict['num_epochs']):

        if stop:
            break
        net.train()
        h = net.initHidden(options_dict['batch_size'])
        h = h.cuda()

        # Training:
        # ---------
        for batch, (y, images) in tqdm(enumerate(trn_loader), desc='Training...', ncols=100):
            if stop:
                break
            itr += 1
            inp_beams, targ, batch_size = traincb.on_batch_begin(y, images)
            h = h.data[:,:batch_size,:].contiguous().cuda()

            opt.zero_grad()
            out, h, alpha = net.forward(inp_beams, images, h)
            out = out.view(-1,out.shape[-1])
            train_loss = criterion(out, targ)  # (pred, target)
            train_loss += alpha_c * ((1. - alpha) ** 2).mean()
            train_loss.backward(retain_graph=True)
            opt.step()
            out = out.view(batch_size,options_dict['out_seq'],options_dict['cb_size'])
            pred_beams = torch.argmax(out,dim=2)
            targ = targ.view(batch_size,options_dict['out_seq'])
            top_1_acc = torch.sum( torch.prod(pred_beams == targ, dim=1, dtype=torch.float) ) / targ.shape[0]
            # pdb.set_trace()
            top_1_score = torch.sum( torch.exp( -torch.norm( pred_beams - targ, 1, dtype=torch.float, dim = 1) 
                                                / options_dict['SIGMA'] / options_dict['out_seq'] ) ) / targ.shape[0]
            
            if np.mod(itr, options_dict['coll_cycle']) == 0:  # Data collection cycle
                running_train_loss.append(train_loss.item())
                running_trn_top_1.append(top_1_acc.item())
                running_trn_top_1_score.append(top_1_score.item())
                train_loss_ind.append(itr)
            if np.mod(itr, options_dict['display_freq']) == 0:  # Display frequency
                # pdb.set_trace()
                print(
                    'Epoch No. {0}--Iteration No. {1}-- Mini-batch loss = {2:10.9f}, Top-1 accuracy = {3:5.4f}, Top-1 score = {4:5.4f}'.format(
                    epoch + 1,
                    itr,
                    train_loss.item(),
                    top_1_acc.item(),
                    top_1_score.item())
                    )

            # Validation:
            # -----------
            if np.mod(itr, options_dict['val_freq']) == 0:  # or epoch + 1 == options_dict['num_epochs']:
                net.eval()
                batch_acc = 0.0
                batch_score = 0.0
                
                with torch.no_grad():
                    for v_batch, (beam, images) in tqdm(enumerate(val_loader), desc='Validating...', ncols=100):
                        inp_beams, targ, _ = traincb.on_batch_begin(beam, images, False)
                        h_val = net.initHidden(beam.shape[0]).cuda()
                        out, h_val, _ = net.forward(inp_beams, images, h_val)
                        pred_beams = torch.argmax(out, dim=2)
                        batch_acc += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
                        # batch_score += torch.sum( torch.exp( - torch.norm( pred_beams - targ, 1, dtype=torch.float) / options_dict['SIGMA'] ))
                        batch_score += torch.sum( torch.exp( -torch.norm( pred_beams - targ, 1, dtype=torch.float, dim = 1) 
                                                / options_dict['SIGMA'] / options_dict['out_seq'] ) )
                    stop = valcb.on_validation_end(net, itr, batch_acc, batch_score)
                net.train()

        current_lr = scheduler.get_lr()[-1]
        scheduler.step()
        new_lr = scheduler.get_lr()[-1]
        if new_lr != current_lr:
            print('Learning rate reduced to {}'.format(new_lr))

    t_end = time.time()
    train_time = (t_end - t_start)/60
    print('Training lasted {0:6.3f} minutes'.format(train_time))
    print('------------------------ Training Done ------------------------')
    running_val_top_1, running_val_top_1_score, val_acc_ind = valcb.get_stat()
    train_info = {'train_loss': running_train_loss,
                  'train_top_1': running_trn_top_1,
                  'train_top_1_score': running_trn_top_1_score,
                  'val_top_1':running_val_top_1,
                  'val_top_1_score': running_val_top_1_score,
                  'train_itr':train_loss_ind,
                  'val_itr':val_acc_ind,
                  'train_time':train_time}

    return [net, options_dict,train_info]
