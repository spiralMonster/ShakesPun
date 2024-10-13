import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.layers import Dense

class AttentionLayer(Layer):
    def __init__(self,seqlen,hidden_dim,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.seqlen=seqlen
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.query=Dense(self.hidden_dim,kernel_initializer='he_uniform',activation='linear',name='attention_query')
        self.key=Dense(self.hidden_dim,kernel_initializer='he_uniform',activation='linear',name='attention_key')
        self.value=Dense(self.hidden_dim,kernel_initializer='he_uniform',activation='linear',name='attention_value')
        self.dense=Dense(self.hidden_dim,kernel_initializer='glorot_uniform',activation='sigmoid',name='attention_dense_layer')
        
        
        
   
        
    def call(self,x):
        query_key=tf.linalg.matmul(self.query(x),tf.transpose(self.key(x),perm=[0,2,1]))
        dense_out=self.dense(query_key)
        value_weights=tf.linalg.matmul(dense_out,tf.transpose(self.value(x),perm=[0,2,1]))
        out=tf.linalg.matmul(value_weights,x)
        return out
        
    def get_config(self):
        config=super().get_config()
        config.update(
        {
          'seqlen':self.seqlen,
          'hidden_dim':self.hidden_dim,
          'out_dim':self.out_dim
        }
        )
        return config
        
        
        
        
        