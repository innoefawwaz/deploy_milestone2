from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import streamlit as st

# App Initialization
app = Flask(__name__)

# Load the Model & prequisites
enc_tokenizer = pickle.load(open('eng_tokenizer.pkl','rb'))
dec_tokenizer = pickle.load(open('nld_tokenizer.pkl','rb'))
answer = pickle.load(open('translated_tokenized_decoder.pkl', 'rb'))
inf_enc_model = tf.keras.models.load_model('model_encoder.h5', compile=False)
inf_dec_model = tf.keras.models.load_model('model_decoder.h5', compile=False)

# Endpoint for Homepage
@app.route("/")
def home():
    return "<h1>It Works!</h1>"

# Endpoint for Prediction
@app.route("/predict", methods=['POST'])
def translation_model():
  args = request.json
  new_data = {
    'user_input': args.get('user_input')
    } 

  input_txt = clean_text(new_data)
  inf = enc_tokenizer.texts_to_sequences([input_txt])
  inf = pad_sequences( inf , maxlen= 14 , padding='post' )

  state_inf = inf_enc_model.predict(inf,verbose=0)

  word = ''
  sentences = []
  target_seq = np.array([[dec_tokenizer.word_index['start']]])
  while True:
    dec_out, h, c = inf_dec_model.predict([target_seq] + state_inf,verbose=0)

    wd_id = np.argmax(dec_out[0][0])
    word = answer[wd_id]
    sentences.append(word)

    target_seq = np.array([[wd_id]])
    state_inf = [h,c]
    
    if word == 'end' or len(sentences)>=15:
      break
    if sentences[-1] == 'end':
      st.write(' '.join(sentences[:,-1]))

    else:
      st.write(' '.join(sentences))

  response = jsonify(
    result = sentences
    )
  return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)