import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add,LSTM,LayerNormalization
from .attention_layer import AttentionLayer

class EncoderLayer(Layer):
    def __init__(self,encoder_seqlen,hidden_dim,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.encoder_seqlen=encoder_seqlen
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.layer_normalization=LayerNormalization()
        self.lstm=LSTM(units=self.out_dim,kernel_initializer='he_uniform',activation='relu',return_sequences=True)
        self.attention_layer=AttentionLayer(seqlen=self.encoder_seqlen,hidden_dim=self.hidden_dim,out_dim=self.out_dim)
        
        
    
    def call(self,embeddings):
        self.stack=embeddings
        out=self.attention_layer(embeddings)
        out=Add()([out,self.stack])
        out=self.layer_normalization(out)
        self.stack=out
        out=self.lstm(out)
        out=Add()([out,self.stack])
        out=self.layer_normalization(out)
        return out
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.encoder_seqlen,self.out_dim)
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'out_dim':self.out_dim,
                'hidden_dim':self.hidden_dim,
                'encoder_seqlen':self.encoder_seqlen
            }
        )
        return config
