import tensorflow as tf

from TCAN_LOCAL.ta import TemporalBlockLayer
import time

'''
    @auther linsu 2021/12/19
'''



class TCANModel(tf.keras.Model):
    def __init__(self,mode:tf.estimator.ModeKeys,params:dict,name:str):
        super(TCANModel,self).__init__(name=name)
        self.training = mode==tf.estimator.ModeKeys.TRAIN
        self.mode = mode
        self.params = params
        self.layers_height = params["tcan_layers_height"]

        self.blocks = [
            TemporalBlockLayer(name="tbl_%d"%i,block_id=i
                               ,head_number=params["tcan_head_number"],key_size=params["tcan_selfattension_keysize"]
                               ,kernal_size=params["tcan_kernal_size"]
                               ,output_size = params["tcan_output_size"]
                               ,include_residule=params["include_residule"]
                                ,conv_dropout =params["conv_dropout"])
            for i in range(self.layers_height)]
        layer_size = list(params.get("hidden_layers"))
        def get_layer_name(id,type_name,outter_name):
            return "{}_{}_{}".format(outter_name,type_name,id)
        def get_layers(outter_name):
            layers = []
            for index,size in enumerate(layer_size):
                layers.append(tf.keras.layers.Dense(size, activation='relu', name=get_layer_name(index,"hidden",outter_name)))
            return layers
        self.ffn_layers = get_layers("tbl_output_net")
        self.output_layer = tf.keras.layers.Dense(params.get("catagory"),name = "tbl_output")
        self.encoder = tf.keras.layers.Dense(params["tcan_output_size"],name="feature_encoder",activation=None)

    def _add_hidden_layer_summary(self, value, tag):
        tf.compat.v1.summary.histogram('%s/activation' % tag, value)

    def call(self,features_dic):
        print("in local now")
        features = features_dic.get("features")


        timestamp = time.time()
        if(self.training):  # disturb the trainning data
            r = tf.keras.backend.truncated_normal(tf.shape(features), mean=0.0, stddev=0.05, dtype=tf.float32,seed=timestamp)
            features = features*(r+1.0)

        features =  tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)
        features = tf.clip_by_value(features,-6,6)
        output = self.encoder(features)  # encoding

        # output= self.addtimevector(output)

        if(self.training):
            output = tf.keras.backend.dropout(output,level=self.params["feature_dropout"])
        history_size=tf.shape(features)[1]
        for block in self.blocks:
            output = block(output,training = self.training)
            output = tf.clip_by_value(output,-6,6)
            self._add_hidden_layer_summary(output, block.name)
        # only the last t is keeped

        _,output_end40 = tf.split(output,[history_size-40,40],axis=1)

        def get_logits(feature, layers):
            for i,layer in enumerate(layers):
                feature = layer(feature)
                # feature = tf.keras.backend.dropout(feature,level=0.1)
                self._add_hidden_layer_summary(feature, layer.name+"_relu")
            return feature
        feature = get_logits(output_end40,self.ffn_layers)
        if(self.training):
            feature = tf.keras.backend.dropout(feature,level=self.params["ffn_dropout"])
        logits = self.output_layer(feature)
        self._add_hidden_layer_summary(logits, self.output_layer.name)
        _,output_1 = tf.split(output_end40,[39,1],axis = 1)
        output_1 = tf.squeeze(output_1,axis =1)
        return logits,output_1


if __name__=="__main__":
    history_vector = tf.expand_dims(tf.range(1,10+1,dtype=tf.float32),-1)
    print(history_vector)
    history_matrix = tf.tile(history_vector,multiples=[1,8-1])
    print(history_matrix)
    time_embedding_head = history_vector/tf.cast(10,tf.float32)
    print(time_embedding_head)
