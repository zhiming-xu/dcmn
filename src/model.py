# !/usr/bin/env python3
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
            self.W = nd.normal(shape=(emb_size, emb_size), ctx=ctx)
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
            self.W = nd.normal(shape=(emb_size, emb_size), ctx=ctx)
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
            self.W_a = nd.normal(shape=(emb_size, emb_size), ctx=ctx)
            self.W_b = nd.normal(shape=(emb_size, emb_size), ctx=ctx)
            self.b = nd.normal(shape=emb_size, ctx=ctx)
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

class MatchOnePair(nn.Block):
    '''
    use the blocks above to complete the model
    '''
    def __init__(self, emb_size, **kwargs):
        super(MatchOnePair, self).__init__(**kwargs)
        with self.name_scope():
            self.birdirmatchbed = BidirMatchEmb(emb_size)
            self.gatedblock = GatedBlock(emb_size)

    def forward(self, emb_a, emb_b):
        S_a, S_b = self.birdirmatchbed(emb_a, emb_b)
        return self.gatedblock(S_a, S_b)

class MatchThreePairs(nn.Block):
    '''
    implement three mathes between (observation1, observation2), (observation1, hypothesis),
    and (observation2, hypothesis)
    '''
    def __init__(self, emb_size, **kwargs):
        super(MatchThreePairs, self).__init__(**kwargs)
        with self.name_scope():
            # match between first ob (before) and second ob (after)
            self.match_o1o2 = MatchOnePair(emb_size)
            # match between first ob (before) and h
            self.match_o1h = MatchOnePair(emb_size)
            # match between second ob (after) and h
            self.match_o2h = MatchOnePair(emb_size)
    
    def forward(self, o1, o2, h):
        # M_* is of dimension (batch_size, emb_size) respectively
        M_o1o2 = self.match_o1o2(o1, o2)
        M_o1h = self.match_o1h(o1, h)
        M_o2h = self.match_o2h(o2, h)
        # the return value of size (batch_size, num_matches*emb_size, 1), num_matches=3 in this case
        return nd.concat(M_o1o2, M_o1h, M_o2h).expand_dims(axis=-1)

class ObjFunc(nn.Block):
    '''
    implement objective function
    '''
    def __init__(self, emb_size, num_matches=3, **kwargs):
        super(ObjFunc, self).__init__(**kwargs)
        with self.name_scope():
            self.V = nd.normal(shape=num_matches*emb_size, ctx=ctx)
            self.expmul = nn.Lambda(
                lambda C: nd.dot(C.transpose(axes=(0, 2, 1)), self.V)
            )

    def forward(self, C):
        # input value is of shape (batch_size, num_matches*emb_size, num_candidates),
        # with num_matches=3, num_candidates=2 in our case
        exp_C = nd.exp(self.expmul(C))
        # L(A_i | P, Q) = -log(exp(V^T C_i) / exp(V^T C))
        L = -nd.log(exp_C / nd.sum(exp_C, axis=-1, keepdims=True))
        return L

class DMCN(nn.Block):
    '''
    wrapper of this whole model
    '''
    def __init__(self, emb_size, num_candidates=2, **kwargs):
        super(DMCN, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = BertEmbedding(ctx=ctx, batch_size=32)
            self.matchthreepairs = [MatchThreePairs(emb_size) for _ in range(num_candidates)]
            self.objfunc = ObjFunc(emb_size)

    def forward(self, *sentences):
        # FIXME: we should not use bert in model, since the dataloader it produces can't be directly
        # use in `model`. `model` should take a batch, an element of a dataloader
        dataloaders = [self.embedding.data_loader(sentence) for sentence in sentences]
        dloader_ob1, dloader_ob2 = dataloaders[:2]
        # each element of this list is a matching matrix of shape (batch_size, num_matches*emb_size),
        # num_matches=3 in our case
        matchmats = [self.matchthreepairs[i](dloader_ob1, dloader_ob2, dataloaders[i+2]) \
                     for i in range(len(dataloaders)-2)]
        # C in original paper, of shape (batch_size, num_matches*emb_size, num_candidates)
        final_representation = nd.concat(*matchmats, dim=-1)
        return self.objfunc(final_representation)
        

