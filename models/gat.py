import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        '''
        hid_units = [8] # numbers of hidden units per each attention head in each layer
        hid_units = [8]表示该GAT只有1层, 该层中有8个unit，
        hid_units的元素作为CNN的output channels

        n_heads = [8, 1] # additional entry for the output layer
        n_heads的元素表示GAT每层的attention抽头的个数


        :param nb_classes:
        :param nb_nodes:
        :param training:
        :param attn_drop:
        :param ffd_drop:
        :param bias_mat:
        :param hid_units: [8]
        :param n_heads: [8, 1]
        :param activation:
        :param residual:
        :return:
        '''
        attns = []
        # GAT的第一层, n_heads是个list, 含义是啥?
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(seq=inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)

        # len(hid_units)表示GAT的层数, 下面这个循环压根就不执行
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            # 对每个hid_units进行遍历
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        out = []
        # GAT的最后一层, 最后输的output channel等于nb_classes, 用于分类
        # GAT最后一层的attention抽头数是1
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        # tf.add_n(inputs) Adds all input tensors element-wise.
        logits = tf.add_n(out) / n_heads[-1]

        return logits
