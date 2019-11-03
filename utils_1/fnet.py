"""
Class for F-network for training the f_theta function
"""

import numpy as np
from tqdm.auto import tqdm
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, padding=1):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding)
        self.bn = batchnorm
        self.b1 = nn.BatchNorm2d(out_filters)
        
    def forward(self, x):
        x = self.c1(x)
        if self.bn: 
            x = self.b1(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, batchnorm=True, activ=F.relu, padding=1):
        '''
        filters: number of input and output filters for convolution layer
        if modifying kernel_size adjust the padding too
        '''
        super(ResidualBlock, self).__init__()
        self.activ = activ
        self.conv1 = ConvBlock(in_filters=filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.conv2 = ConvBlock(in_filters=filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
    
    def forward(self,x):
        x_mod = self.activ(self.conv1(x))
        x_mod = self.conv2(x_mod)
        x_mod = self.activ(x_mod+x)
        return x_mod


class PolicyHead(nn.Module):
    def __init__(self, filters=2, kernel_size=1, batchnorm=True, activ=F.relu, board_size=13, res_filters=256, padding=0):
        super(PolicyHead, self).__init__()
        self.conv = ConvBlock(in_filters=res_filters, out_filters=filters, 
                              kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.activ = activ
        self.fc_infeatures = filters*board_size*board_size
        self.fc = nn.Linear(in_features=self.fc_infeatures, out_features=board_size*board_size+1)
    
    def forward(self,x):
        x = self.activ(self.conv(x))
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        # print(x)
        return torch.softmax(x,1)


class ValueHead(nn.Module):
    def __init__(self, filters=1, kernel_size=1, batchnorm=True, activ=F.relu, board_size=13, res_filters=256, padding=0):
        super(ValueHead, self).__init__()
        self.conv = ConvBlock(in_filters=res_filters, out_filters=filters, 
                              kernel_size=kernel_size, batchnorm=batchnorm,  padding=padding)
        self.activ = activ
        self.fc_infeatures = filters*board_size*board_size
        self.fc1 = nn.Linear(in_features=self.fc_infeatures, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
    
    def forward(self,x):
        x = self.activ(self.conv(x))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.tanh(x)


class AlphaNeural(nn.Module):
    def __init__(self, res_blocks, board_size):
        super(AlphaNeural, self).__init__()
        self.res_blocks = res_blocks
        self.board_size = board_size
        self.input_stack = 17
        self.res_filters = 256
        
        self.conv1 = ConvBlock(in_filters=self.input_stack, out_filters=self.res_filters)
        for i in range(self.res_blocks):
            self.add_module("ResBlock"+str(i), ResidualBlock(filters=self.res_filters))
        self.policy_head = PolicyHead(board_size=self.board_size, res_filters=self.res_filters)
        self.value_head = ValueHead(board_size=self.board_size, res_filters=self.res_filters)
    
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        for i in range(self.res_blocks):
            x = self._modules["ResBlock"+str(i)](x)
        # print(x.shape)

        action = self.policy_head(x)
        # print(action)
        value = self.value_head(x)
        return action, value


class NeuralTrainer():
    def __init__(self, res_blocks, board_size, lr=1e-3, epochs=10, batch_size=64):
        self.net = AlphaNeural(res_blocks=res_blocks, board_size=board_size)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_crit = torch.nn.MSELoss()
        
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()
    
    def train(self, examples, logging=True, log_file=None):
        '''examples is list containing (state, policy, value)'''
        
        opt = torch.optim.Adam(self.net.parameters())
        self.net.train()
        states = torch.Tensor([x[0] for x in examples])
        policies = torch.Tensor([x[1] for x in examples])
        values = torch.Tensor([x[2] for x in examples])
        train_dataset = TensorDataset(states, policies, values)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        policy_loss = 0
        value_loss = 0
        for epoch in range(self.epochs):
            for s,pi,v in train_loader:
                if self.cuda_flag:
                    s,pi,v = s.contiguous().cuda(), pi.contiguous().cuda(), v.contiguous().cuda()
                
                pred_pi, pred_v = self.net(s)
                
                pi_loss =  -torch.sum(pred_pi*pi)/pi.numel() # this loss calculates how closely the two policies match
                val_loss = self.value_crit(pred_v.view(-1), v)
                
                policy_loss+=pi_loss.item()
                value_loss+=val_loss.item()
                
                total_loss = pi_loss+val_loss
                
                opt.zero_grad()
                total_loss.backward()
                opt.step()
        if logging:
            print("Policy Loss:{} Value Loss:{}".format(policy_loss, value_loss))
        if log_file is not None:
            with open(log_file, 'a') as f:
                timestamp = str(datetime.datetime.now()).split('.')[0]
                f.write("{} | Policy Loss:{} Value Loss:{}\n".format(timestamp, policy_loss, value_loss))

        return policy_loss, value_loss
    
    def predict(self, board):
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.cuda_flag: 
            board = board.contiguous().cuda()
        board = board.unsqueeze(0)
        self.net.eval()
        
        with torch.no_grad():
            pi, v = self.net(board)
        
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def save_model(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path))


# class FNet:
#     def __init__ (self, board_size, load_net=None):
#         # Initialize the network
#         self.board_size = board_size
#         self.pass_action = board_size * board_size
#         self.total_actions = board_size * board_size + 1
#         self.board_shape = (board_size, board_size)

#         # ... other stuff
#         if load_net:
#             self.load_network(load_net)

#     def load_network(self, infile):
#         # Load the network from a file
#         with open(infile, 'rb') as f:
#             print ('Loaded')

#     def dump_network(self, outfile):
#         # Dump the network in a file for future use
#         with open(outfile, 'wb') as f:
#             print ('Dumped')

#     def foward_pass(self, state):
#         # Doing forward pass through the FNet
#         assert (state.shape == self.board_shape)

#         value = np.random.rand() * 2 - 1
#         policy = [np.random.rand() for i in range(0, self.total_actions)]
#         policy = np.array(policy / sum(policy))

#         return value, policy

#     def update_net(self, batch):
#         # Batch will be an array of (s, pi, r) tuples
#         # s : State - 2D numpy array of shape self.board_size
#         # pi : Policy values - 1D numpy array of size self.actions
#         # r : +1, 0, -1 depending on whether white won

#         pass
        