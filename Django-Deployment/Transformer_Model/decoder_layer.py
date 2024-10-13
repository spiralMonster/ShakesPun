import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Add,LayerNormalization
from .attention_layer import AttentionLayer

class DecoderLayer(Model):
    def __init__(self,decoder_seqlen,hidden_dim1,hidden_dim2,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.decoder_seqlen=decoder_seqlen
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.out_dim=out_dim
        self.layer_normalization=LayerNormalization()
        self.lstm=LSTM(units=self.out_dim,kernel_initializer='he_uniform',activation='relu',return_sequences=True)
        self.attention_layer1=AttentionLayer(seqlen=self.decoder_seqlen,hidden_dim=self.hidden_dim1,out_dim=self.out_dim)
        self.attention_layer2=AttentionLayer(seqlen=self.decoder_seqlen,hidden_dim=self.hidden_dim2,out_dim=self.out_dim)
        self.stack=None
    
    
        
    def call(self,embeddings,encoder_output):
        self.stack=embeddings
        out=self.attention_layer1(embeddings)
        out=Add()([out,self.stack])
        out=self.layer_normalization(out)
        self.stack=out
        out= self.attention_layer2(out)
        out=Add()([out,encoder_output])
        out=Add()([out,self.stack])
        out=self.layer_normalization(out)
        self.stack=out
        out=self.lstm(out)
        out=Add()([out,self.stack])
        out=self.layer_normalization(out)
        return out
    
        
    def compute_output_shape(self,input_shape):
        return(input_shape[0],self.decoder_seqlen,self.out_dim)
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'decoder_seqlen':self.decoder_seqlen,
                'out_dim':self.out_dim,
                'hidden_dim2':self.hidden_dim2,
                'hidden_dim1':self.hidden_dim1
            }
        )
        return config
    


        