import tensorflow as tf
from keras.layers import Layer

class ShapeAligner(Layer):
    def __init__(self,embedding_dim,**kwargs):
        super().__init__(**kwargs)
        self.embedding_dim=embedding_dim
        
    
        
    def call(self,x):
        paddings=tf.constant([[0,0],[0,1],[0,0]])
        return tf.pad(x,paddings,mode='CONSTANT',constant_values=0.0)
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'embedding_dim':self.embedding_dim   
            }
        )
        return config