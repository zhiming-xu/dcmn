import mxnet as mx, gluonnlp as nlp
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd, gluon
from bert.embedding import BertEmbedding
import pandas as pd, numpy as np
import logging

np.random.seed(9102)
mx.random.seed(8064)

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

try:
    nd.ones(10, ctx=mx.gpu())
    ctx = mx.gpu()
except Exception as e:
    logger.warning('No GPU device found with exception {}. Will use CPU instead'.format(e))
    ctx = mx.cpu()

class AttentionWeightMatrix(nn.Block):
    '''
    implement G^qa = softmax(H^p W H^aT)
    '''
    def __init__(self, emb_size, **kwargs):
        super(AttentionWeightMatrix, self).__init__(**kwargs)
        with self.name_scope():
            self.W = nd.random.normal(shape=(emb_size, emb_size), ctx=ctx)
            # emb_a: batch_size*seq_len_a*emb_size, emb_b: batch_size*seq_len_b*emb_size
            # self.W: emb_size*emb_size
            # After the evaluation, the shape is batch_size*seq_len_a*emb_size_b
            self.attmat = nn.Lambda(lambda *emb: nd.softmax(
                nd.batch_dot(
                    nd.dot(emb[0], self.W),
                    nd.transpose(emb[1], axes=(0, 2, 1))
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
            self.W = nd.random.normal(shape=(emb_size, emb_size), ctx=ctx)
            self.cal_E = nn.Lambda(lambda *args: nd.dot(
                        args[0], nd.transpose(args[1], axes=(1, 0, 2)))
                    )
            self.cal_ReLU = nn.Lambda(lambda E: nd.relu(nd.dot(E, self.W)))

    def forward(self, G, H):
        return self.cal_ReLU(self.cal_E(G, H))

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
            self.W_a = nd.random.normal(shape=(emb_size, emb_size), ctx=ctx)
            self.W_b = nd.random.normal(shape=(emb_size, emb_size), ctx=ctx)
            self.b = nd.random.normal(shape=emb_size, ctx=ctx)
            self.maxpooling = nn.GlobalMaxPool1D(layout='NWC')
            self.gate_rate = nn.Lambda(
                lambda *args: nd.sigmoid(nd.dot(args[0], self.W_a) + \
                                         nd.dot(args[1], self.W_b) + self.b)
            )
            self.gate_output = nn.Lambda(
                lambda *args: args[2] * args[0] + (1 - args[2]) * args[1]
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
    def __init__(self, emb_size, **kwargs):
        super(MatchPair, self).__init__(**kwargs)
        self.birdirmatchbed = BidirMatchEmb(emb_size)
        self.gatedblock = GatedBlock(emb_size)

    def forward(self, emb_a, emb_b):
        S_a, S_b = self.birdirmatchbed(emb_a, emb_b)
        return self.gatedblock(S_a, S_b)

class ObjFunc(nn.Block):
    '''
    implement objective function
    '''
    def __init__(self, emb_size, **kwargs):
        self.V = nd.random.normal(shape=emb_size, ctx=ctx)

    def forward(self, C):
        


