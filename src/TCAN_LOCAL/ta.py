import math

import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import linalg
from tensorflow.python.ops.gen_array_ops import matrix_band_part

'''
    @auther linsu 2021/12/19
'''

class TemporalSelfAttensionLayer(tf.keras.layers.Layer):
    def __init__(self,name,dtype=tf.float32,key_size:int=128,output_size:int=396):
        super(TemporalSelfAttensionLayer,self).__init__(name=name,dtype=dtype)
        self.key_size = key_size
        self.q_dense = tf.keras.layers.Dense(key_size)
        self.k_dense = tf.keras.layers.Dense(key_size)
        self.v_dense = tf.keras.layers.Dense(output_size)
    def call(self, inputs,*args, **kwargs):
        q = self.q_dense(inputs) #[batch_size,seq_len,key_size]
        k = self.k_dense(inputs)#[batch_size,seq_len,key_size]
        v = self.v_dense(inputs) #[batch_size,seq_len,output_size]
        simlarity = tf.matmul(q,tf.transpose(k,[0,2,1])) #[batch_size,seq_len,seq_len]
        simlarity = simlarity/math.sqrt(self.key_size)

        padding = -1e7
        simlarity =  matrix_band_part(simlarity,num_upper=3,num_lower=3)
        mask = tf.where(tf.abs(simlarity)>0.0,0.0,1.0)
        simlarity = simlarity+padding*mask
        simlarity_softmax = tf.nn.softmax(simlarity,axis=-1) #[batch_size,seq_len,seq_len] ignored
        # v = [batch_size,seq_len,output_size]
        output = tf.matmul(simlarity_softmax,v) #[batch_size,seq_len, outsize]
        return output



class TemporalBlockLayer(tf.keras.layers.Layer):
    def __init__(self,name:str="TA",block_id:int=0, head_number:int=1,key_size:int=128,kernal_size:int = 4
                 ,output_size:int=396,include_residule=True,conv_dropout = 0.2,dtype=tf.float32):
        super(TemporalBlockLayer,self).__init__(name=name,dtype=dtype)
        self.head_number = float(head_number)
        self.include_residule = include_residule
        head_output_size = max(128,output_size/head_number) if head_number>1 else output_size
        key_size = max(128,output_size/head_number) if head_number>1 else key_size
        self.attension_unit =[TemporalSelfAttensionLayer("%s_head%d" % (name,i),key_size=key_size,output_size=head_output_size)
                               for i in range(head_number)]  #up,down,even
        self.output_size = output_size
        self.output_encoder = None if head_number==1 else tf.keras.layers.Dense(self.output_size)
        # 1,4,16,64
        #range = 64*4= 256
        # dilation_rate = int(math.pow(3,block_id))
        # 27*4 = 108
        #1,6,36
        # 1,3,9,27,91
        dilation_rate =int(math.pow(kernal_size,block_id))
        #tf.keras.backend.relu
        self.conv = tf.keras.layers.Conv1D(output_size,kernel_size=kernal_size,dilation_rate=dilation_rate
                                           ,padding="causal",activation=tf.keras.backend.relu,name=name+"_conv",use_bias=True)
        self.conv_dropout = conv_dropout
        # self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1,epsilon=1e-7)


    def call(self, inputs, **kwargs):
        training = kwargs.get("training")
        output_array = [unit(inputs,training)
            for unit in self.attension_unit]
        if self.output_encoder is None:
            out =  output_array[0]
        else:
            output_raw = tf.concat( output_array,axis=-1)
            out = self.output_encoder(output_raw)
        conv_res = self.conv(out)
        result = conv_res+inputs
        # result=self.norm_layer(result)

        result = tf.nn.leaky_relu(result)
        if(training):
            result = tf.keras.backend.dropout(result,level=self.conv_dropout)
        return result



def main(_):
    input = np.array([[[7, 7, 7, 7,7,7,7],              # Input shape: (2, 4, 4)
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7]],
                      [[7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7],
                       [7, 7, 7, 7,7,7,7]]])
    # seq_len = 4
    # batch_size = 2
    #
    # k = (2,3)
    # max_diag_len = min(seq_len + min(k[1], 0), seq_len + min(-k[0], 0))
    # num_diags = k[1] - k[0] + 1
    # diagonal = tf.ones([batch_size,num_diags,max_diag_len])*(-float('inf'))
    #
    # diagonal= tf.ones([batch_size,4])*(-float('inf'))
    # simlarity =  tf.linalg.set_diag(diagonal,diagonal=input,k=1)
    padding = tf.cast(tf.ones_like(input),tf.float32)*(-1e7)
    simlarity =  tf.cast(matrix_band_part(input,num_upper=0,num_lower=2),tf.float32)
    mask = tf.where(tf.abs(simlarity)>0,0.0,1.0)
    simlarity = simlarity+padding*mask
    print(simlarity.numpy())
    simlarity_softmax = tf.nn.softmax(simlarity,axis=-1) #[batch_size,seq_len,seq_len]

    print(simlarity_softmax.numpy())

if __name__ == "__main__":
    main(None)