import numpy as np
import tensorflow as tf
import sys

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    '''
    这是attention的实现,每个attention抽头都要调用该函数
    :param seq: 模型的输入是ftr_in：(batch_size, nb_nodes, ft_size) -> (1, 2708, 1433), 1433相当于论文中的F
    :param out_sz: 就是论文中的F'
    :param bias_mat: 2708 * 2708
    :param activation:
    :param in_drop:
    :param coef_drop:
    :param residual:
    :return:
    '''
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # 下面的conv1d的数学过程: inputs中的channel都各自乘同一个数, 类似于线性变化, 这就是1d卷积
        # inputs的格式默认是(batch, length, channels)
        # 对于GAT最后一层seq_fts:(batch, length, out_sz) -> (batch, length, 8)
        # 对于GAT最后一层seq_fts:(batch, length, out_sz) -> (1, 2708, 8)
        # seq是论文中的h=(h1,h2,h3,...,hN), seq_fts是论文中的h'=(h'1,h'2,...,h'N)
        # 这里表示不带attention的图卷积操作
        seq_fts = tf.layers.conv1d(inputs=seq, filters=out_sz, kernel_size=1, use_bias=False)

        # 下面要计算attention mechanism a,   怎么找出邻居节点?
        # simplest self-attention possible
        # 此处的tf.layers.conv1d是论文中提到的 'a is a single-layer feedforward neural network' 的 neural network
        # 这个操作相当于把每个节点特征的各个维度进行加权求和,最终每个节点特征由一个标量表示
        f_1 = tf.layers.conv1d(inputs=seq_fts, filters=1, kernel_size=1)
        f_2 = tf.layers.conv1d(inputs=seq_fts, filters=1, kernel_size=1)
        # 下面这步是啥意思?
        # logits是个1*2708*2708的矩阵, 不考虑第0维, 只看第1维和第2维构成的矩阵: logits_ij表示特征i和特征j相加的结果
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # 默认在axis=-1上进行softmax
        coefs = tf.nn.softmax(tf.nn.leaky_relu(features=logits, alpha=0.2) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))
        # 下面的f_1, f_2都是2703*2703的
        f_1 = adj_mat*f_1 # 格外注意： 这个乘法操作是element-wise的, 不是矩阵乘法！  广播机制
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

