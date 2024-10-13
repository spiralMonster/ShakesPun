import numpy as np
import pandas as pd
import os
import tensorflow as tf
import PyPDF2
import re
import string
import random
from tensorflow.keras.preprocessing.text import Tokenizer

class DataPreparation:
    def __init__(self,pdf_path,tokenizer_file_path,input_length):
        self.pdf_path=pdf_path
        self.input_length=input_length
        self.tokenizer_file_path=tokenizer_file_path
        
    def data_extraction(self):
        data=""
        with open(self.pdf_path,"rb") as file:
            reader=PyPDF2.PdfReader(file)
            num_pages=len(reader.pages)
            for page_no in range(num_pages):
                page=reader.pages[page_no]
                text=page.extract_text()
                data+=text
                data+=' '
        return data
        
    def data_cleaning(self):
        data=self.data_extraction()
        data=re.sub(r'\s\s+',' ',data)
        table1=str.maketrans('','',string.punctuation)
        table2=str.maketrans('','','0123456789')
        cleaned_data=""
        for word in data.split(" "):
            if word.isalpha():
                word=word.translate(table1)
                word=word.translate(table2)
                word=word.lower()
                cleaned_data+=word
                cleaned_data+=" "
        return cleaned_data
        
    def data_generation(self):
        data=self.data_cleaning()
        tokenizer=Tokenizer(oov_token="OOV")
        data=data.split(" ")
        num_of_words=len(data)
        X=[]
        Y=[]
        window_size=self.input_length-1
        for ind in range(num_of_words-self.input_length):
            X.append(" ".join(data[ind:ind+window_size]).strip())
            Y.append(data[ind+window_size])
            
        tokenizer.fit_on_texts(X)
        X=tokenizer.texts_to_sequences(X)
        Y=tokenizer.texts_to_sequences(Y)
        word_index=tokenizer.word_index
        word_index_new={}
        data=list(zip(X,Y))
        random.shuffle(data)
        X,Y=zip(*data)
        X=list(X)
        Y=list(Y)
        
        tokenizer_json=tokenizer.to_json()
        with open(os.path.join(self.tokenizer_file_path, '../Trained Model/tokenizer.json'), 'w') as file:
            file.write(tokenizer_json)
            
        for key,value in word_index.items():
            word_index_new[key]=value-1
            
        return X,Y,word_index_new
    
        
        
        
                