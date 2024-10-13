import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input,Concatenate,Dense,LSTM
from transformers_embedding_layer import EmbeddingLayer
from attention_layer import AttentionLayer
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from shape_aligner import ShapeAligner
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from cosine_annealing_scheduler import CosineAnnealingScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split


class Transformer:
    
    def __init__(self,num_of_encoder_decoder_stack,encoder_seqlen,decoder_seqlen,encoder_hidden_dim,
                 decoder_hidden_dim,embedding_dim,embedding_path,dense_units,batch_size):
        self.num_of_encoder_decoder_stack=num_of_encoder_decoder_stack
        self.embedding_dim=embedding_dim
        self.encoder_seqlen=encoder_seqlen
        self.decoder_seqlen=decoder_seqlen
        self.encoder_hidden_dim=encoder_hidden_dim #list
        self.decoder_hidden_dim=decoder_hidden_dim #list
        self.embedding_path=embedding_path
        self.dense_units=dense_units #List-> only upto last second layer
        self.batch_size=batch_size
        
        
    def fit(self,X,Y,word_index):
        self.encoder_inp=np.concatenate((X,Y),axis=1)
        self.decoder_inp=np.array(X)
        self.decoder_Y=np.array(Y)
        self.word_index=word_index
        self.vocab_size=len(word_index)
        self.embedding_matrix_generation()
    
    
    def embedding_matrix_generation(self):
        if self.embedding_path is not None:
            embedding_matrix=np.zeros(shape=(self.vocab_size,self.embedding_dim),dtype="float32")
            embed_index={}
            with open(self.embedding_path,"r",encoding="utf-8") as file:
                for line in file:
                    vector=line.split()
                    word=vector[0]
                    values=np.asarray(vector[1:],dtype='float32')
                    embed_index[word]=values
                    
            for word in self.word_index.keys():
                if word in embed_index.keys():
                    embedding_matrix[self.word_index[word]]=embed_index[word]
        else:
            embedding_matrix=np.random.normal(shape=(len(self.word_index)+1,self.embedding_dim),mean=0.0,stddev=1.0)
            
        self.embedding_matrix=embedding_matrix
        
        
    def build(self):
        inp_encoder=Input(shape=(self.encoder_seqlen,),dtype=tf.float32,name='encoder_input')
        inp_decoder=Input(shape=(self.decoder_seqlen,),dtype=tf.float32,name='decoder_input')
        print(inp_encoder.shape)
        print(inp_decoder.shape)
        
        #embedding matrices:
        embed_matrix=self.embedding_matrix
        
        
        #embeddings:
        embed_encoder=EmbeddingLayer(seqlen=self.encoder_seqlen,
                                     embedding_dim=self.embedding_dim,
                                     embedding_matrix=embed_matrix,
                                     vocab_size=self.vocab_size,
                                     name='enocder_embedding')(inp_encoder)
        
        embed_decoder=EmbeddingLayer(seqlen=self.decoder_seqlen,
                                     embedding_dim=self.embedding_dim,
                                     embedding_matrix=embed_matrix,
                                     vocab_size=self.vocab_size,
                                     name='decoder_embedding')(inp_decoder)
        print(embed_encoder)
        print(embed_decoder)
        
        out_decoder=ShapeAligner(embedding_dim=self.embedding_dim,
                                 name='Shape_Aligner')(embed_decoder)
        print(out_decoder.shape)
        
        out_encoder=embed_encoder
        self.decoder_seqlen=self.encoder_seqlen
        
        for ind in range(self.num_of_encoder_decoder_stack):
            
            out_encoder=EncoderLayer(encoder_seqlen=self.encoder_seqlen,
                                     hidden_dim=self.encoder_hidden_dim[0],
                                     out_dim=self.embedding_dim,
                                     name=f'encoder_stack_{ind+1}')(out_encoder)
            
            
            out_decoder=DecoderLayer(decoder_seqlen=self.decoder_seqlen,
                                     hidden_dim1=self.encoder_hidden_dim[0],
                                     hidden_dim2=self.encoder_hidden_dim[1],
                                     out_dim=self.embedding_dim,
                                     name=f'decoder_stack_{ind+1}')(out_decoder,out_encoder)
            
        out_decoder=LSTM(units=self.dense_units[0],kernel_initializer='he_uniform',activation='relu',return_sequences=False)(out_decoder)
        for ind in range(len(self.dense_units)-1):
            out_decoder=Dense(units=self.dense_units[ind+1],kernel_initializer='he_uniform',activation='relu')(out_decoder)
            
        
        out_decoder=Dense(units=self.vocab_size,kernel_initializer='glorot_uniform',activation='softmax')(out_decoder)
        
        Transformer_model=Model(inputs=[inp_encoder,inp_decoder],outputs=[out_decoder])
        
        return Transformer_model
        
    def train(self,initial_learning_rate,epochs,filepath):
        
        transformer_model=self.build()
        transformer_model.summary()
        
        opt=Adam(learning_rate=initial_learning_rate,beta_1=0.96,beta_2=0.98)
        transformer_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        callback1= CosineAnnealingScheduler(initial_learning_rate=initial_learning_rate,total_epochs=epochs)
        
        

        history=transformer_model.fit(x=[self.encoder_inp,self.decoder_inp],
                                      y=self.decoder_Y,
                                      epochs=epochs,
                                      callbacks=[callback1],
                                      batch_size=self.batch_size
                                      )

                                
        return history
    
        
        
        
        
        
        
    
            
            
            
        



        
       
        




        
        
        
        
        
        
    