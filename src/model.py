import mxnet as mx, gluonnlp as nlp
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd, gluon
from bert.embedding import BertEmbedding
import pandas as pd, numpy as np
import logging

np.random.seed(9102)
mx.random.seed(8064)


try:
    nd.ones(10, ctx=mx.gpu())
except Exception as e:
    

class AttentionWeightMatrix(nn.Block):
    '''
    implement G^qa = softmax(H^p W H^aT)
    '''
    def __init__(self, emb_size, **kwargs):
        super(AttentionWeightMatrix, self).__init__(**kwargs)
        with self.name_scope():
            self.W = nd.random.normal(shape=(emb_size, emb_size), ctx=mx.gpu())
            # emb_a: batch_size*seq_len_a*emb_size, emb_b: batch_size*seq_len_b*emb_size
            # self.W: emb_size*emb_size
            # After the evaluation, the shape is batch_size*seq_len_a*emb_size_b
            self.attmat = nn.Lambda(lambda emb_a, emb_b: nd.softmax(
                nd.batch_dot(
                    nd.dot(emb_a, self.W),
                    nd.transpose(emb_b, axes=(0, 2, 1))
                ), axis=1)
            )

    def forward(self, emb_a, emb_b):
        return self.attmat(emb_a, emb_b)
        
class SoftAlignment(nn.Block):
    '''
    implement S_a/b = RELU(G_ab emb_a/(G_ab)^T emb_b, W_a/b)
    '''
    def __init__(self, emb_size, **kwargs):
        super(SoftAlignment, self).__init__(**kwargs)
        with self.name_scope():
            # the parameter matrix W
            self.W = nd.random.normal(shape=(emb_size, emb_size), ctx=mx.gpu())
            self.softalign = nn.Sequential(
                nn.Lambda(lambda G, H: nd.dot(
                    G, nd.transpose(H, axes=(1, 0, 2)))
                ),
                nn.Lambda(lambda E: nd.relu(nd.dot(E, self.W)))
            )

    def forward(self, G, H):
        return self.softalign(G, H) 

class BidirMatchEmb(nn.Block):
    '''
    for bidirectional matching representation
    '''
    def __init__(self, emb_size, **kwargs):
        '''
        init function, we will leave the bert embedding for now, suppose embeddings
        are already available
        '''
        super(BidirMatchEmb, self).__init__(**kwargs)
        with self.name_scope():
            # self.embedding_layer = BertEmbedding # FIXME: this is incomplete
            self.attweightmat = AttentionWeightMatrix(emb_size)
            self.softalign_a = SoftAlignment(emb_size)
            self.softalign_b = SoftAlignment(emb_size)

    def forward(self, emb_a, emb_b):
        # G = softmax(emb_a W emb_b), (batch_size, len_a, len_b)
        G = self.attweightmat(emb_a, emb_b)
        # S_a = ReLU(G emb_b W_a), (batch_size, len_a, emb_size)
        S_a = self.softalign_a(G, emb_b)
        # S_b = ReLU(G^T -> batch_size*len_b*len_a, emb_a, W_b), (batch_size, len_b, emb_size)
        S_b = self.softalign_b(G.transpose(axes=(0, 2, 1)), emb_a)
        return S_a, S_b

class GatedBlock(nn.Block):
    '''
    implement gated mechanism, input is S_a/S_b
    '''
    def __init__(self, emb_size, **kwargs):
        super(GatedBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.W_a = nd.random.normal(shape=(emb_size, emb_size), ctx=mx.gpu())
            self.W_b = nd.random.normal(shape=(emb_size, emb_size), ctx=mx.gpu())
            self.b = nd.random.normal(shape=(emb_size, 1), ctx=mx.gpu())
            self.maxpooling = nn.GlobalMaxPool1D(layout='NWC')
            self.gate_rate = nn.Lambda(
                lambda M_a, M_b: nd.sigmoid(nd.dot(M_a, self.W_a) + \
                                 nd.dot(M_b, self.W_b) + self.b)
            )
            self.gate_output = nn.Lambda(
                lambda M_a, M_b, g: g * M_a + (1 - g) * M_b
            )

    def forward(self, S_a, S_b):
        M_a = self.maxpooling(S_a).flatten(dim=1)
        M_b = self.maxpooling(S_b).flatten(dim=1)
        gr = self.gate_rate(M_a, M_b)
        return self.gate_output(M_a, M_b, gr)

class MatchPair(nn.Block):
    '''
    use the blocks above to complete the model
    '''
    def __init__(self, emb_size):
        pass

