import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

class PositionalEmbeddingLayer(Layer):
    def __init__(self,seqlen,embedding_dim,**kwargs):
        super().__init__(**kwargs)
        self.embedding_dim=embedding_dim
        self.seqlen=seqlen
        self.position_matrix_generation()
        self.position_embedding=self.add_variable(shape=(self.seqlen,self.embedding_dim),
                                                  dtype=tf.float32,
                                                  trainable=True,
                                                  initializer=Constant(self.position_matrix),
                                                  name='positional_embedding')
        
        
    def position_matrix_generation(self):
        
        position_matrix=np.zeros(shape=(self.seqlen,self.embedding_dim),dtype='float32')
        n=10000
        
        for k in range(self.seqlen):
            for i in np.arange(int(self.embedding_dim/2)):
                denom=np.power(n,2*i/self.embedding_dim)
                position_matrix[k,2*i]=np.sin(k/denom)
                position_matrix[k,2*i+1]=np.cos(k/denom)
                
        self.position_matrix=position_matrix
    
    
        
    def call(self,x):
        batch_size=tf.shape(x)[0]
        seqlen=tf.shape(x)[1]
        out=tf.broadcast_to(self.position_embedding,(batch_size,seqlen,self.embedding_dim))
        return out
        
    def get_config(self):
        config=super().get_config()
        
        config.update(
            {
                'embedding_dim':self.embedding_dim,
                'position_matrix':self.position_matrix,
                'seqlen':self.seqlen
            }
        )
        return config
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.embedding_dim)
        
        