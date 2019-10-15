#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:00:12 2019

@author: carsault
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class ModelFamily(nn.Module):
    def __init__(self):
        super(ModelFamily, self).__init__()
        self.models = nn.ModuleDict()
        self.decim = []

    def addModel(self, model, decim):
        self.models[decim] = model
        self.decim.append(decim)
        
    def forward(self, x, args, bornInf, bornSup):
        out = []
        i = 0
        for d in self.decim:
            if d != str(1) :
                data = x[:,bornInf[int(d)]:bornSup[int(d)],:].to(args.device)
                out.append(self.models[d].encoder(data))
            i += 1
        out = torch.cat(out, 1)
        data = x[:,bornInf[1]:bornSup[1],:].to(args.device)
        #print(data)
        y = self.models["1"](data,out)
        return y
    
    
class ModelFamilySum(nn.Module):
    def __init__(self):
        super(ModelFamilySum, self).__init__()
        self.models = nn.ModuleDict()
        self.decim = []

    def addModel(self, model, decim):
        self.models[decim] = model
        self.decim.append(decim)
        
    def forward(self, x, args):
        out = []
        i = 0
        for d in self.decim:
            if d != str(1) :
                data = x[i].to(args.device)
                data = self.models[d](data)
                data = data.repeat(1,int(d),1)
                data = data.div(int(d))
                out.append(data)
            i += 1
        data = x[0].to(args.device)
        out.append(self.models["1"](data))
        out = torch.stack(out)
        y = torch.sum(out, dim = 0)
        return y
    
    
class InOutModel(nn.Module):
    def __init__(self, encoder, decoder):
       super(InOutModel, self).__init__()
       self.encoder = encoder
       self.decoder = decoder
       
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
    
class FinalModel(nn.Module):
    def __init__(self, encoder, decoder):
       super(FinalModel, self).__init__()
       self.encoder = encoder
       self.decoder = decoder
       
    def forward(self, x, out):
        y = self.encoder(x)
        y = self.decoder(y, out)
        return y
       
class EncoderMLP(nn.Module):
    def __init__(self, lenSeq, n_categories, n_hidden, n_latent, decimRatio, n_layer = 1, dropRatio = 0.5):
        super(EncoderMLP, self).__init__()
        self.fc1 = nn.Linear(int(lenSeq * n_categories / decimRatio), n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.Linear(n_hidden, n_latent)
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.decimRatio = decimRatio
        self.lenSeq = lenSeq
        self.n_layer = n_layer
    def forward(self, x):
        x = x.view(-1, int(self.lenSeq * self.n_categories/ self.decimRatio))
        x = F.relu(self.fc1(x))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x = self.fc3(x)
        return x

class dilatConv(nn.Module):
    def __init__(self, lenSeq, lenPred, n_categories, latent):
        super(dilatConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 250, 2, 1, dilation = (2,1))
        self.conv2 = nn.Conv2d(250, 25, 2, 1, dilation = (4,1))
        self.fc1 = nn.Linear(1150, 500)
        self.fc2 = nn.Linear(500, latent)
        self.lenPred = lenPred
        self.lenSeq = lenSeq
        self.n_categories = n_categories
        self.latent = latent

    def forward(self, x):
        x = x.view(-1, 1, self.lenSeq, self.n_categories)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1150)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def name(self):
        return "dilatConv"
    
class DecoderMLP(nn.Module):
    def __init__(self, lenPred, n_categories, n_hidden, n_latent, decimRatio, n_layer = 1, dropRatio = 0.5):
        super(DecoderMLP, self).__init__()
        self.fc1 = nn.Linear(n_latent , n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.Linear(n_hidden, int(lenPred * n_categories / decimRatio))
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.decimRatio = decimRatio
        self.lenPred = lenPred
        self.n_layer = n_layer
    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x = self.fc3(x)
        x = x.view(-1, int(self.lenPred / self.decimRatio), self.n_categories)
        if self.decimRatio == 1 :
            x = nn.Softmax(dim=2)(x)
        else:
            x = F.relu(x)
        return x
    
class DecoderFinal(nn.Module):
    def __init__(self, lenSeq, lenPred, n_categories, n_hidden, n_latent, n_layer = 1, dropRatio = 0.5):
        super(DecoderFinal, self).__init__()
        self.fc1 = nn.Linear(n_latent , n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.Linear(n_hidden, lenPred * n_categories)
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.lenPred = lenPred
        self.n_layer = n_layer
    def forward(self, x, out):
        x = torch.cat((x,out), 1) 
        x = F.relu(self.fc1(x))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x = self.fc3(x)
        x = x.view(-1, self.lenPred, self.n_categories)
        x = nn.Softmax(dim=2)(x)
        return x

#%%
#%%
class VAEModelFamily(nn.Module):
    def __init__(self):
        super(VAEModelFamily, self).__init__()
        self.models = nn.ModuleDict()
        self.decim = []

    def addModel(self, model, decim):
        self.models[decim] = model
        self.decim.append(decim)     
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 
    def forward(self, x, args):
        out = []
        i = 0
        for d in self.decim:
            if d != str(1) :
                data = x[i].to(args.device)
                out1, out2 = self.models[d].encoder(data)
                out.append(self.reparametrize(out1,out2))
            i += 1
        out = torch.cat(out, 1)
        data = x[0].to(args.device)
        #print(data)
        y = self.models["1"](data,out)
        return y
    
class VAEInOutModel(nn.Module):
    def __init__(self, encoder, decoder):
       super(VAEInOutModel, self).__init__()
       self.encoder = encoder
       self.decoder = decoder
       
    def forward(self, x):
        y1, y2 = self.encoder(x)
        y = self.decoder(y1, y2)
        return y
    
class VAEFinalModel(nn.Module):
    def __init__(self, encoder, decoder):
       super(VAEFinalModel, self).__init__()
       self.encoder = encoder
       self.decoder = decoder
       
    def forward(self, x, out):
        y1, y2 = self.encoder(x)
        y = self.decoder(y1, y2, out)
        return y
       
class VAEEncoderMLP(nn.Module):
    def __init__(self, lenSeq, n_categories, n_hidden, n_latent, decimRatio, n_layer = 1, dropRatio = 0.5):
        super(VAEEncoderMLP, self).__init__()
        self.fc1 = nn.Linear(int(lenSeq * n_categories / decimRatio), n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc31 = nn.Linear(n_hidden, n_latent)
        self.fc32 = nn.Linear(n_hidden, n_latent)
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.decimRatio = decimRatio
        self.lenSeq = lenSeq
        self.n_layer = n_layer
    def forward(self, x):
        x = x.view(-1, int(self.lenSeq * self.n_categories/ self.decimRatio))
        x = F.relu(self.fc1(x))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x1 = self.fc31(x)
        x2 = self.fc32(x)
        return x1, x2
    
class VAEDecoderMLP(nn.Module):
    def __init__(self, lenPred, n_categories, n_hidden, n_latent, decimRatio, n_layer = 1, dropRatio = 0.5):
        super(VAEDecoderMLP, self).__init__()
        self.fc1 = nn.Linear(n_latent , n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.Linear(n_hidden, int(lenPred * n_categories / decimRatio))
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.decimRatio = decimRatio
        self.lenPred = lenPred
        self.n_layer = n_layer
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std        
    def forward(self, x1, x2):
        z = self.reparametrize(x1, x2)
        x = F.relu(self.fc1(z))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x = self.fc3(x)
        x = x.view(-1, int(self.lenPred / self.decimRatio), self.n_categories)
        if self.decimRatio == 1 :
            x = nn.Softmax(dim=2)(x)
        return x, x1, x2
    
class VAEDecoderFinal(nn.Module):
    def __init__(self, lenSeq, lenPred, n_categories, n_hidden, n_latent, n_layer = 1, dropRatio = 0.5):
        super(VAEDecoderFinal, self).__init__()
        self.fc1 = nn.Linear(n_latent , n_hidden)
        self.fc2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.Linear(n_hidden, lenPred * n_categories)
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.lenPred = lenPred
        self.n_layer = n_layer
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x1, x2, out):
        x = self.reparametrize(x1, x2)
        x = torch.cat([x,out], 1) 
        x = F.relu(self.fc1(x))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.fc2[i](x))
        x = self.fc3(x)
        x = x.view(-1, self.lenPred, self.n_categories)
        x = nn.Softmax(dim=2)(x)
        return x, x1, x2
#%%
class MLPNet(nn.Module):
    def __init__(self, lenSeq, lenPred, n_categories):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(lenSeq * n_categories, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, lenPred * n_categories)
        self.lenPred = lenPred
        self.n_categories = n_categories
    def forward(self, x):
        x = x.view(-1, 16*25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.lenPred, self.n_categories)
        x = nn.Softmax(dim=2)(x)
        return x
    
#%%
class LeNet(nn.Module):
    def __init__(self, lenSeq, lenPred, n_categories):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(8*17*50, 500)
        self.fc2 = nn.Linear(500, lenPred * n_categories)
        self.lenPred = lenPred
        self.lenSeq = lenSeq
        self.n_categories = n_categories

    def forward(self, x):
        x = x.view(-1, 1, self.lenSeq, self.n_categories)
        x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8*17*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.lenPred, self.n_categories)
        x = nn.Softmax(dim=2)(x)
        return x
    
    def name(self):
        return "LeNet"
    
#%%

#%%
n_inputs = 25
n_hidden = 128
batch_size = 500
lenSeq = 16
n_categories=25
class MockupModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=n_inputs,    # 45, see the data definition
                hidden_size=n_hidden,  # Can vary
                num_layers = 3,
                dropout = 0.6, #0.6
                batch_first = True
            ),
            'linear': nn.Linear(
                in_features=n_hidden,
                out_features=n_categories)
        })
            
    def forward(self, x):

        # From [batches, seqs, seq len, features]
        # to [seq len, batch data, features]
        # Data is fed to the LSTM
        out, _ = self.model['lstm'](x)
        #print(f'lstm output={out.size()}')

        # From [seq len, batch, num_directions * hidden_size]
        # to [batches, seqs, seq_len,prediction]
        out = out.view(batch_size, lenSeq, -1)
        #print(f'transformed output={out.size()}')

        # Data is fed to the Linear layer
        out = self.model['linear'](out)
        #print(f'linear output={out.size()}')

        # The prediction utilizing the whole sequence is the last one
        #y_pred = nn.Softmax()(y_pred)
        y_pred = out[:, -1]
        y_pred = nn.Softmax()(y_pred)
        
        #print(f'y_pred={y_pred.size()}')

        return y_pred
#%%    
class MockupModelMask(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=n_inputs,    # 45, see the data definition
                hidden_size=n_hidden,  # Can vary
                num_layers = 3,
                dropout = 0.6, #0.6
                batch_first = True
            ),
            'linear': nn.Linear(
                in_features=n_hidden,
                out_features=n_categories)
        })
            
    def forward(self, x, nbZero, mask = False):

        # From [batches, seqs, seq len, features]
        # to [seq len, batch data, features]
        if mask == True:
            for i in range(x.size()[0]):
                for j in range(nbZero):
                    x[i][randint(0,15)] = torch.zeros(n_inputs)
        # Data is fed to the LSTM
        out, _ = self.model['lstm'](x)
        #print(f'lstm output={out.size()}')

        # From [seq len, batch, num_directions * hidden_size]
        # to [batches, seqs, seq_len,prediction]
        out = out.view(batch_size, lenSeq, -1)
        #print(f'transformed output={out.size()}')

        # Data is fed to the Linear layer
        out = self.model['linear'](out)
        #print(f'linear output={out.size()}')

        # The prediction utilizing the whole sequence is the last one
        #y_pred = nn.Softmax()(y_pred)
        y_pred = out[:, -1]
        y_pred = nn.Softmax()(y_pred)
        
        #print(f'y_pred={y_pred.size()}')
#%%
class ResBlock(nn.Module):
    def __init__(self, dim, dim_res=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim_res, 3, 1, 1),
            nn.BatchNorm2d(dim_res),
            nn.ReLU(True),
            nn.Conv2d(dim_res, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return x + self.block(x)
    
class View1(nn.Module):
   def __init__(self):
        super(View1, self).__init__()
   
   def forward(self, x):
        return x.view(-1,16*24)
    
class View2(nn.Module):
   def __init__(self):
        super(View2, self).__init__()
   
   def forward(self, x):
        return x.view(-1,1,16,25) #make it with lenPred
#%%
# Construct encoders and decoders for different types
def construct_enc_dec(input_dim, dim, embed_dim = 64):
    encoder, decoder = None, None
    # Image data
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, int(dim / 2), 4, 2, 1),
        #nn.BatchNorm2d(dim),
        nn.ReLU(True),
        nn.Conv2d(int(dim / 2), dim, 4, 2, 1),
        #nn.BatchNorm2d(dim),
        nn.ReLU(True),
        nn.Conv2d(dim, dim, 3, 1, 1),
        ResBlock(dim),
        ResBlock(dim),
        nn.Conv2d(dim, embed_dim, 1)
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(embed_dim, dim, 3, 1, 1),
        ResBlock(dim),
        ResBlock(dim),
        nn.ConvTranspose2d(dim, int(dim / 2), 4, 2, 1),
        #nn.BatchNorm2d(dim),
        nn.ReLU(True),
        nn.ConvTranspose2d(int(dim / 2), input_dim, 4, 2, 1),
        View1(),
        nn.Linear(16*24,16*25), #make it with lenPred
        View2()
        #nn.Tanh()
    )
    return encoder, decoder

#%% Seq 2 Seq from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html -> see also attention is page
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.device = device
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    

    
