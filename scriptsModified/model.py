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

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        att1 = self.encoder_att(encoder_out)  # (batch_size, seq, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (num_layer , batch_size, attention_dim)
        att = self.full_att(self.relu(att1.unsqueeze(0) + att2.unsqueeze(2))).squeeze(3)  # (num_layer, batch_size, seq)
        alpha = self.softmax(att)  # (num_layer,batch_size, seq)
        attention_weighted_encoding = (encoder_out.unsqueeze(0) * alpha.unsqueeze(3)).sum(dim=0)  # (batch_size, seq, encoder_dim)

        return attention_weighted_encoding, alpha

# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 beam_dim,
                 img_dim,
                 hid_dim,
                 inp_seq,
                 attention_dim,
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

        # attention network
        self.attention = Attention(img_dim, hid_dim, attention_dim)  

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

        images, alpha = self.attention(images,h)

        x = torch.cat((beams,images),2)
        # pdb.set_trace()
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1*self.out_seq:,:])
        y = self.classifier(out)
        # y = self.softmax(out)
        return [y, h, alpha]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )

