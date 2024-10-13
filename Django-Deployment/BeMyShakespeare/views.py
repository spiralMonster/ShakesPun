from django.shortcuts import render
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
import Transformer_Model
from Transformer_Model.attention_layer import AttentionLayer
from Transformer_Model.transformers_embedding_layer import EmbeddingLayer
from Transformer_Model.positional_layer_embedding import PositionalEmbeddingLayer
from Transformer_Model.shape_aligner import ShapeAligner
from Transformer_Model.cosine_annealing_scheduler import CosineAnnealingScheduler
from Transformer_Model.encoder_layer import EncoderLayer
from Transformer_Model.decoder_layer import DecoderLayer
import json
# TF_ENABLE_ONEDNN_OPTS=0
def home_page(request,*args,**kwargs):
    if request.POST:
        with open("./Trained Model/index_word.json","r") as file:
            index_word=json.load(file)
        with open("./Trained Model/tokenizer.json",'r') as file:
            tokenizer_json=file.read()
        tokenizer=tokenizer_from_json(tokenizer_json)
        text=request.POST['text_input']
        tokenized_text=tokenizer.texts_to_sequences([text])
        with CustomObjectScope({'AttentionLayer':AttentionLayer,
                              'EmbeddingLayer':EmbeddingLayer,
                              'PositionalEmbeddingLayer':PositionalEmbeddingLayer,
                              'ShapeAligner':ShapeAligner,
                              'EncoderLayer':EncoderLayer,
                              'DecoderLayer':DecoderLayer,
                              'CosineAnnealingScgeduler':CosineAnnealingScheduler}):

            model=load_model("./Trained Model/model.h5")
        len_text=len(tokenized_text[0])
        max_words=50
        if len_text<9:
            len_mask=9-len_text
            mask=len_mask*[0]
            input=mask.copy()
            input.extend(tokenized_text[0])
        else:
            input=tokenized_text[0].copy()

        gen_text=text

        while len(gen_text.split(" ")) !=max_words:
            decoder_inp = input[-9:]
            decoder_inp=[float(inst) for inst in decoder_inp]
            decoder_inp = np.array(decoder_inp)
            decoder_inp = np.expand_dims(decoder_inp, axis=0)
            encoder_inp = input[-9:]
            encoder_inp.append(0)
            encoder_inp = np.array(encoder_inp)
            encoder_inp = np.expand_dims(encoder_inp, axis=0)
            pred=model.predict([encoder_inp,decoder_inp],batch_size=1)
            temp=0.07
            scaled_logits=pred[0]/temp
            exp_logits=np.exp(scaled_logits)
            probabilities=exp_logits/np.sum(exp_logits)
            pred=np.random.choice(len(probabilities),p=probabilities)
            input.append(pred)
            pred=str(pred)
            word=index_word[pred]
            gen_text+=" "
            gen_text+=word
        print(gen_text)
        context={'text':gen_text}

    return render(request,"bemyshakespeare_home.html",{})
