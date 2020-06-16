import torch
import torch.nn as nn
import torchvision
import os
import pdb

# class Encoder(nn.Module):
#     """
#     Encoder.
#     """

#     def __init__(self, embed_size):
#         super(Encoder, self).__init__()  
# #         resnet = torchvision.models.resnet18(pretrained=True)
#         resnet = torchvision.models.resnet18()
#         resnet.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
#         # Remove linear and pool layers (since we're not doing classification)
#         modules = list(resnet.children())[:-1]      # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, images):
#         """
#         Forward propagation.
#         :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
#         :return: encoded images
#         """
#         with torch.no_grad():
#             features = self.resnet(images)
#         features = features.reshape(features.size(0), -1)
#         features = self.bn(self.linear(features))
#         return features


# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 beam_dim,
                 img_dim,
                 hid_dim,
                 inp_seq,
                 out_dim,
                 out_seq,
                 num_layers,
                 drop_prob=0.3
                 ):
        super(RecNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_seq = out_seq
        # self.orig_dim = orig_dim
        self.num_layers = num_layers
        self.inp_seq = inp_seq

        # Define layers
        self.gru = nn.GRU(beam_dim+img_dim,hid_dim,num_layers,batch_first=True,dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()

        #ResNet-18
        resnet = torchvision.models.resnet18()
        #resnet = torchvision.models.resnet18(pretrained=True)
        resnet.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, img_dim)
        self.bn = nn.BatchNorm1d(img_dim, momentum=0.01)
        # self.softmax = nn.Softmax(dim=1)--> Softmax is implicitly implemented into the cross entropy loss

    def forward(self,beams,images,h):
        # x = self.image_base(x)
        # ResNet
        shape = images.shape
        images = images.view(shape[0]*self.inp_seq,shape[2],shape[3],shape[4]).float()
        images = images.cuda()
        images = self.resnet(images)
        images = images.reshape(images.size(0), -1)
        images = self.bn(self.linear(images))
        images = images.view(shape[0],self.inp_seq,-1)

        x = torch.cat((beams,images),2)
        # pdb.set_trace()
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1*self.out_seq:,:])
        y = self.classifier(out)
        # y = self.softmax(out)
        return [y, h]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )

