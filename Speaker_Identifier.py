from keras.layers import Input, Subtract, Dense, Lambda
from keras.models import Model
import keras.backend as K

def build_siamese_network(encoder, input_shape):
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    
    # `encoder` is any predefined network that maps a single sample 
    # into an embedding space.
    # `encoder` should take an input with shape (None,) + input_shape
    # and produce an output with shape (None, embedding_dim). 
    # None indicates the batch dimension.
    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)
    
    # Here I calculate the eucliden distance between the two encoded
    # samples though other distances can be used
    embedded_distance = Subtract()([encoded_1, encoded_2])
    embedded_distance = Lambda(
        lambda x: K.sqrt(K.mean(K.square(x), axis=-1,keepdims=True))
    )(embedded_distance)
    
    # Add a dense+sigmoid layer here in order to use per-pair, binary 
    # similar/dissimilar labels
    output = Dense(1, activation='sigmoid')(embedded_distance)
    
    siamese = Model(inputs=[input_1, input_2], outputs=output)
    
    return siamese