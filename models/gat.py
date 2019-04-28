import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        # GAT的第一层, n_heads是个list, 含义是啥?
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        # hid_units应该表示GAT的层数
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
        # GAT的最后一层
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
