import numpy as np

import functions

class Linear:
    '''
    Applies a linear transformation to the incoming data:
        y = np.dot(x, W.T) + b 
    '''
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = np.random.randn(self.out_features, self.in_features) / np.sqrt(self.in_features / 2)
        self.dW = np.zeros_like(self.W)
        self.cache_W = 0 # cache for rms_prop
        
        self.b = np.zeros(self.out_features)
        self.db = np.zeros_like(self.b)
        self.cache_b = 0
        
    def __call__(self, x):
        '''
        Arguments:
            x: in_features of shape (batch_size, in_features)
           
        Returns:
            Z: out_features of shape (batch_size, out_features)
        '''
        
        self.x = x
        self.Z = np.dot(self.x, self.W.T) + self.b
        return self.Z
    
    def backward(self, dtop):
        '''
        Arguments:
            dtop: gradient from the top of shape (out_features, batch_size)
            
        Returns
            dx: shape (batch_size, in_features)
            dw: shape (out_features, in_features)
        '''    
        # += only affects lstm (feedback connections) 
        # It does not affect vanilla mlp because gradients are set to zero after
        # every forward and backward pass
        self.db += np.sum(dtop, axis=0)
        
        self.dx = np.dot(dtop, self.W)
        self.dW += np.dot(dtop.T, self.x)
        
        return self.dx
    
    def update(self, lr, decay_rate):
        ''' 
        RMS prop 
        '''
        self.cache_W = (decay_rate * self.cache_W) + ((1 - decay_rate) * (self.dW**2))
        self.W += - lr * self.dW / (np.sqrt(self.cache_W) + 1e-7)
        self.dW *= 0
        
        self.cache_b = (decay_rate * self.cache_b) + ((1 - decay_rate) * (self.db**2))
        self.b += - lr * self.db / (np.sqrt(self.cache_b) + 1e-7)
        self.db *= 0
        

class LSTM:
    def __init__(self, INPUT, HIDDEN):
        self.INPUT = INPUT # Its actually input features. ie. emb_dim
        self.HIDDEN = HIDDEN

        self.fcf = Linear(self.INPUT, self.HIDDEN)
        self.fci = Linear(self.INPUT, self.HIDDEN)
        self.fcc = Linear(self.INPUT, self.HIDDEN)
        self.fco = Linear(self.INPUT, self.HIDDEN)
        
    def forward(self, input_val):
        self.Xt = input_val
        
        batch_num = input_val.shape[1]
        
        self.caches = []
        self.states = []
        self.states.append([np.zeros([batch_num, self.HIDDEN]), 
                            np.zeros([batch_num, self.HIDDEN])])
        
        for x in input_val:
            c_prev, h_prev = self.states[-1]
        
            x = np.column_stack([x, h_prev])
            
            hf = functions.sigmoid(self.fcf(x))
            hi = functions.sigmoid(self.fci(x))
            hc = functions.sigmoid(self.fcc(x))
            ho = functions.tanh(self.fco(x))
            
            self.c = hf * c_prev + hi * hc
            self.h = ho * functions.tanh(self.c)
        
            self.states.append([self.c, self.h])
            self.caches.append([x, hf, hi, ho, hc])
            
        self.c, self.h = self.states[-1]
        return self.h, self.caches[-1][3]
        
    
    def backward(self, dtop):       
        dc_next = np.zeros_like(self.c)
        dh_next = np.zeros_like(self.h)
        
        dx_total = []
        for t in range(self.Xt.shape[0]):
            c, h = self.states[-t-1]
            c_prev, h_prev = self.states[-t-2]
    
            x, hf, hi, ho, hc = self.caches[-t-1]
            
            tc = functions.tanh(c)
            dh = dtop + dh_next
            
            dc = dh * ho * functions.tanh_backward(tc)
            dc = dc + dc_next
            
            dho = dh * tc 
            dho = dho * functions.sigmoid_backward(ho)
            
            dhf = dc * c_prev 
            dhf = dhf * functions.sigmoid_backward(hf)
            
            dhi = dc * hc 
            dhi = dhi * functions.sigmoid_backward(hi)
            
            dhc = dc * hi 
            dhc = dhc * functions.tanh_backward(hc)
    
            dXf = self.fcf.backward(dhf)
            dXi = self.fcf.backward(dhi)
            dXc = self.fcf.backward(dhc)
            dXo = self.fcf.backward(dho)
    
            dX = dXf + dXi + dXo + dXc
            
            dc_next = hf * dc
            dh_next = dX[:, -self.HIDDEN:]
            
            dx_time_step = dX[:, :-self.HIDDEN]
            dx_total.append(dx_time_step)

        dx_total = np.stack(dx_total, axis=0)
        return dx_total
   
    
class Embedding:
    def __init__(self, vocab_size, emb_dim):
        '''
        Create an embedding matrix of size (vocab_size+1, emb_dim)
        The first row is reserved for <UNKNOWN> token, and hence is zero.
        '''
        self.vocab_size = vocab_size
        
        self.W = np.random.randn(self.vocab_size+1, emb_dim) / np.sqrt(emb_dim)
        self.W[0] = 0
        
        self.cache = 0
        
    def __call__(self, x):
        '''
        Arguments: x - dataset of size (batch_size, seq_len)
        Returns: x_emb - dataset with token number replaced by corresponding 
                    embedding of size (batch_size, seq_len, emb_dim)
        '''
        self.x = x
        x_emb = self.W[self.x]
        return x_emb
    
    def sort_grads(self, dtop):
        '''
        Assign grads to corresponding W_emb
        Arguments: dtop: same shape as x_emb (batch_size, seq_len, emb_dim)
        '''
        self.dW = []
        for i in range(self.vocab_size+1):
            i_grad = dtop[self.x == i]
            i_grad = np.mean(i_grad, axis=0)
            self.dW.append(i_grad)
        self.dW = np.stack(self.dW)
        self.dW[0] = 0 # No grad for <UNK> token 
        np.nan_to_num(self.dW, copy=False)
        
        assert self.W.shape == self.dW.shape

    def update(self, lr, decay_rate):
        '''
        RMS Prop
        '''
        self.cache = (decay_rate * self.cache) + ((1 - decay_rate) * (self.dW**2))
        self.W += - lr * self.dW / (np.sqrt(self.cache) + 1e-7)
       
        
class Model:
    emb_dim = 50
    mlp_out = 2
    
    def __init__(self, args):
        self.lstm_hidden = args.lstm_hidden
        self.mlp_hidden = args.mlp_hidden
         
        self.emb     = Embedding(args.vocab_size, Model.emb_dim)
        
        self.lstm = LSTM(Model.emb_dim + self.lstm_hidden, self.lstm_hidden)
        
        self.fc1     = Linear(self.lstm_hidden, self.mlp_hidden)
        self.relu1   = functions.Relu()
        self.drop1   = functions.Dropout(dropout_percent=0.4)
        self.fc2     = Linear(self.mlp_hidden, Model.mlp_out)
        self.softmax = functions.Softmax()
        
    def __call__(self, x, dropout=False):
        x = self.emb(x)
        x = np.transpose(x, [1,0,2])
        x, _ = self.lstm.forward(x)
        x = self.fc1(x)
        x = self.relu1(x)
        if dropout: x = self.drop1(x)
        x = self.fc2(x)
        x = self.softmax(x, dim=1)
        return x
    
    def backward(self, y, dropout=False):
        dtop = self.softmax.backward(y)
        dtop = self.fc2.backward(dtop)
        if dropout: dtop = self.drop1.backward(dtop)
        dtop = self.relu1.backward(dtop)
        dtop = self.fc1.backward(dtop)
        dtop = self.lstm.backward(dtop)
        dtop = np.transpose(dtop, [1,0,2])
        self.emb.sort_grads(dtop)
        
    def update(self, lr, decay_rate):
        self.emb.update(lr, decay_rate)
        
        self.lstm.fcf.update(lr, decay_rate)
        self.lstm.fci.update(lr, decay_rate)
        self.lstm.fcc.update(lr, decay_rate)
        self.lstm.fco.update(lr, decay_rate)
        
        self.fc1.update(lr, decay_rate)
        self.fc2.update(lr, decay_rate)
        
    def reset_rms_cache(self):
        self.emb.cache = 0

        self.lstm.fcf.cache_W, self.lstm.fcf.cache_b = 0, 0
        self.lstm.fci.cache_W, self.lstm.fci.cache_b = 0, 0
        self.lstm.fcc.cache_W, self.lstm.fcc.cache_b = 0, 0
        self.lstm.fco.cache_W, self.lstm.fco.cache_b = 0, 0
        
        self.fc1.cache_W, self.fc1.cache_b = 0, 0
        self.fc2.cache_W, self.fc2.cache_b = 0, 0