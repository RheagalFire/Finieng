import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Failed')
    # Invalid device or cannot modify virtual devices once initialized.
    pass

#import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input


latent_dim=256
model = tf.keras.models.load_model('./Enc_files/Encoder_Decoder_model.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim),name='input_3')
decoder_state_input_c = Input(shape=(latent_dim),name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


#encoder_model=tf.keras.models.load_model("./Enc_files/encoder_model")
#decoder_model=tf.keras.models.load_model("./Enc_files/decoder_model")
input_token_index=pickle.load(open('./Enc_files/input_token_index.pickle','rb'))
output_token_index=pickle.load(open('./Enc_files/output_token_index.pickle','rb'))
max_encoder_len=18
max_decoder_len=66
encoder_tokens=pickle.load(open('./Enc_files/encoder_tokens.pickle','rb'))
decoder_tokens=pickle.load(open('./Enc_files/decoder_tokens.pickle','rb'))

def get_seq(input_text):
    print('Translating: ',input_text)
    encoder_input=np.zeros((1,max_encoder_len,encoder_tokens),dtype='float32')
    for i,char in enumerate(input_text):
        encoder_input[0,i,input_token_index[char]]=1
        print('input token index: ',input_token_index[char])
    encoder_input[0,i+1:,input_token_index[' ']]=1
    return encoder_input

def Translate(input_text):
    encoder_input=get_seq(input_text)
    states_values=encoder_model.predict(encoder_input)
    target_seq = np.zeros((1, 1, decoder_tokens))
    target_seq[0, 0, output_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentance=''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_values)
        sampled_token_index=np.argmax(output_tokens[0,-1,:])
        print('sampled token: ',sampled_token_index)
        char=list(output_token_index.keys())[list(output_token_index.values()).index(sampled_token_index)]
        print('char_predicted: ',char[0])
        decoded_sentance+=char[0]
        if (char[0] == '\n' or len(decoded_sentance) > max_decoder_len):
            stop_condition = True
        target_seq = np.zeros((1, 1,decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_values=[h,c]
    return decoded_sentance[:-1]

if __name__=='__main__':
    print(Translate('Go'))