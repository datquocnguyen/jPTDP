"""
https://raw.githubusercontent.com/bplank/bilstm-aux/master/src/lib/mnnl.py
"""
import dynet
import numpy as np

import sys

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")

class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict_sequence(self, inputs):
        return [self.network_builder(x) for x in inputs]


class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return s_init.transduce(inputs)

class BiRNNSequencePredictor(SequencePredictor):
    """ a bidirectional RNN (LSTM/GRU) """
    def __init__(self, f_builder, b_builder):
        self.f_builder = f_builder
        self.b_builder = b_builder

    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.f_builder.initial_state()
        b_init = self.b_builder.initial_state()
        forward_sequence = f_init.transduce(f_inputs)
        backward_sequence = b_init.transduce(reversed(b_inputs))
        return forward_sequence, backward_sequence 
        

class Layer:
    """ Class for affine layer transformation or two-layer MLP """
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh, mlp=0, mlp_activation=dynet.rectify):
        # if mlp > 0, add a hidden layer of that dimension
        self.act = activation
        self.mlp = mlp
        if mlp:
            print('>>> use mlp with dim {} ({})<<<'.format(mlp, mlp_activation))
            mlp_dim = mlp
            self.mlp_activation = mlp_activation
            self.W_mlp = model.add_parameters((mlp_dim, in_dim))
            self.b_mlp = model.add_parameters((mlp_dim))
        else:
            mlp_dim = in_dim
        self.W = model.add_parameters((output_dim, mlp_dim))
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        if self.mlp:
            W_mlp = dynet.parameter(self.W_mlp)
            b_mlp = dynet.parameter(self.b_mlp)
            act = self.mlp_activation
            x_in = act(W_mlp * x + b_mlp)
        else:
            x_in = x
        # from params to expressions
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)
        return self.act(W*x_in + b)
