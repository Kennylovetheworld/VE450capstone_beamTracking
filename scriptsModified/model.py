import torch
import torch.nn as nn

# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 inp_dim,
                 hid_dim,
                 out_dim,
                 out_seq,
                 num_layers,
                 img_dim,
                 cnn_channels,
                 drop_prob=0.3
                 ):
        super(RecNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_seq = out_seq
        # self.orig_dim = orig_dim
        self.num_layers = num_layers
        self.cnn_channels = cnn_channels

        # Define layers
        self.gru = nn.GRU(inp_dim,hid_dim,num_layers,batch_first=True,dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()
        self.image_base = nn.Sequential(
            nn.Conv2d(3,cnn_channels[0],3,padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels[0],cnn_channels[1],3,padding=1),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.softmax = nn.Softmax(dim=1)--> Softmax is implicitly implemented into the cross entropy loss

    def forward(self,x,h):
        # x = self.image_base(x)
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1*self.out_seq:,:])
        y = self.classifier(out)
        # y = self.softmax(out)
        return [y, h]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )

