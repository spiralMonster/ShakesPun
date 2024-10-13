import tensorflow as tf
from tensorflow.keras.layers import Layer,Embedding,Add,Reshape
from tensorflow.keras.initializers import Constant
from .positional_layer_embedding import PositionalEmbeddingLayer

class EmbeddingLayer(Layer):
    def __init__(self,seqlen,embedding_dim,embedding_matrix,vocab_size,**kwargs):
        super().__init__()
        self.seqlen=seqlen
        self.embedding_dim=embedding_dim
        self.embed_matrix=embedding_matrix
        self.vocab_size=vocab_size
        self.embedding_layer=Embedding(input_dim=self.vocab_size,
                                       output_dim=self.embedding_dim,
                                       embeddings_initializer=Constant(self.embed_matrix),
                                       trainable=True)
        self.position_embedding=PositionalEmbeddingLayer(seqlen=self.seqlen,embedding_dim=self.embedding_dim)
        
        
    def call(self,inp):
        x=self.embedding_layer(inp)
        y=self.position_embedding(inp)
        x=Add()([x,y])
        return x
        
    def get_config(self):
        config=super().get_config()
        config.update(
        {
          'embedding_dim':self.embedding_dim,
          'embedding_matrix':self.embed_matrix,
          'vocab_size':self.vocab_size,
          'seqlen':self.seqlen
        }
      )
        return config
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.embedding_dim)
        
    
