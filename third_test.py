import numpy as np
import tensorflow as tf

def Multi_Heads_DotProductAttention(batchsize, sequence_length, d_model, num_heads):
    X = np.random.random((batchsize, sequence_length, d_model)) #生成随机的X张量作为输出
    print(X)
    #生成权重张量，用X来分别乘以他们得到Q，K，V
    w_Q = np.random.random((batchsize, d_model, d_model))
    w_K = np.random.random((batchsize, d_model, d_model))
    w_V = np.random.random((batchsize, d_model, d_model))

    Q = np.matmul(X, w_Q)
    K = np.matmul(X, w_K)
    V = np.matmul(X, w_V)

    #将Q、K和V张量进行形状变换，使其包含多个注意力头。通过函数将Q、K和V张量的形状变为(batchsize, sequence_length, num_heads, depth)
    depth = d_model // num_heads #用于计算每个注意力头的深度大小（即特征维度的子空间维度）
    Q = np.reshape(Q, [batchsize, sequence_length, num_heads, depth])
    K = np.reshape(K, [batchsize, sequence_length, num_heads, depth])
    V = np.reshape(V, [batchsize, sequence_length, num_heads, depth])


    #将注意力头的维度放在最后一个维度上,转置num_heads, sequence_length为了满足多头自注意力机制在注意力计算中的并行、交互性和兼容性需求
    Q = tf.transpose(Q, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]
    K = tf.transpose(K, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]
    V = tf.transpose(V, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]

    #转置K， 做QK的乘积运算
    K = tf.transpose(K, perm=[0, 1, 3, 2])  # [batchsize, num_heads, d_model//num_heads, sequence_length]
    dk = tf.cast(tf.shape(K)[-1], tf.float32) #相似度除以他的根号可以减小方差
    attention_matrix = np.matmul(tf.nn.softmax(np.matmul(Q, K) / np.sqrt(dk), axis=-1), V) 
    attention_matrix = tf.transpose(attention_matrix, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention = tf.reshape(attention_matrix,
                                  (batchsize, -1, d_model))  # (batch_size, seq_len_q, d_model) 和输入a维度一致，接下来就可以做残差连接
    return concat_attention
print("---------------------------------------------")
print(Multi_Heads_DotProductAttention(batchsize=16, sequence_length=3, d_model=8, num_heads=2))