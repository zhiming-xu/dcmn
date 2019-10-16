# !/usr/bin/env python3
import mxnet as mx, gluonnlp as nlp
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd, gluon
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
            self.W = self.params.get(
                'att_weight_matrix_W', shape=(emb_size, emb_size)
            )

    def forward(self, emb_a, emb_b):
        # emb_a: batch_size*seq_len_a*emb_size, emb_b: batch_size*seq_len_b*emb_size
        # self.W: emb_size*emb_size
        # After the evaluation, the shape is batch_size*seq_len_a*emb_size_b
        return nd.softmax(nd.batch_dot(nd.dot(emb_a, self.W.data()), \
                                       nd.transpose(emb_b, axes=(0, 2, 1))), axis=1)
        
class SoftAlignment(nn.Block):
    '''
    implement S_a/b = RELU(G_ab emb_a/(G_ab)^T emb_b, W_a/b)
    '''
    def __init__(self, emb_size, **kwargs):
        super(SoftAlignment, self).__init__(**kwargs)
        with self.name_scope():
            # the parameter matrix W
            # self.W = nd.normal(scale=.1, shape=(emb_size, emb_size), ctx=ctx).attach_grad()
            self.W = self.params.get(
                'soft_align_W', shape=(emb_size, emb_size)
            )

    def forward(self, G, emb):
        E = nd.dot(G, nd.transpose(emb, axes=(1, 0, 2)))
        relu_EW = nd.relu(nd.dot(E, self.W.data()))
        return relu_EW

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
            self.W_a = self.params.get(
                'gated_block_W_a', shape=(emb_size, emb_size)
            )
            self.W_b = self.params.get(
                'gated_block_W_b', shape=(emb_size, emb_size)
            )
            self.b = self.params.get(
                'gated_block_b', shape=emb_size
            )
            self.maxpooling = nn.GlobalMaxPool1D(layout='NWC')

    def forward(self, S_a, S_b):
        M_a = self.maxpooling(S_a).flatten(dim=1)
        M_b = self.maxpooling(S_b).flatten(dim=1)
        gate_rate = nd.relu(nd.dot(M_a, self.W_a.data()) + \
                            nd.dot(M_b, self.W_b.data()) + self.b.data())
        gated_output = gate_rate * M_a + (1 - gate_rate) * M_b
        return gated_output

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
            self.V = self.params.get(
                'objective_function_V', shape=num_matches*emb_size
            )

    def forward(self, C):
        # input value is of shape (batch_size, num_matches*emb_size, num_candidates),
        # with num_matches=3, num_candidates=2 in our case
        exp_C = nd.exp(nd.dot(C.transpose(axes=(0, 2, 1)), self.V.data()))
        # L(A_i | P, Q) = -log(exp(V^T C_i) / exp(V^T C))
        L = -nd.log(exp_C / nd.sum(exp_C, axis=-1, keepdims=True))
        return L

class DMCN(nn.Block):
    '''
    wrapper of this whole model
    '''
    def __init__(self, emb_size=768, num_candidates=2, **kwargs):
        super(DMCN, self).__init__(**kwargs)
        with self.name_scope():
            self.matchthreepairs = [MatchThreePairs(emb_size) for _ in range(num_candidates)]
            for block in self.matchthreepairs:
                self.register_child(block)
            self.objfunc = ObjFunc(emb_size)

    def forward(self, inputs):
        '''
        inputs: obs1, obs2, hyp1, hyp2, (could be more hyps)
        '''
        # each element of this list is a matching matrix of shape (batch_size, num_matches*emb_size),
        # num_matches=3 in our case
        matchmats = [self.matchthreepairs[i](inputs[0], inputs[1], inputs[i+2]) \
                     for i in range(len(inputs)-2)]
        # C in original paper, of shape (batch_size, num_matches*emb_size, num_candidates)
        final_representation = nd.concat(*matchmats, dim=-1)
        return self.objfunc(final_representation)

        