from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, Input, Model
from tensorflow.keras.layers import Lambda, multiply, concatenate, Dense, Embedding, Reshape
from tensorflow.keras import initializers

userlayers = [512,64]
itemlayers = [1024,64]
layers =[512,256,128,64]
num_users = n_plylst
num_items = n_songs

def get_model():
    dmf_num_layer = len(userlayers) #Number of layers in the DMF
    mlp_num_layer = len(layers) #Number of layers in the MLP

    # Input variables
    user_rating = Input(shape=(num_items,), dtype='int32', name='user_input')
    item_rating = Input(shape=(num_users,), dtype='int32', name='item_input')

    # DMF part
    userlayer = Dense(userlayers[0],  activation="linear" , name='user_layer0')
    itemlayer = Dense(itemlayers[0], activation="linear" , name='item_layer0')
    dmf_user_latent = userlayer(user_rating)
    dmf_item_latent = itemlayer(item_rating)
    for idx in range(1, dmf_num_layer):
        userlayer = Dense(userlayers[idx],  activation='relu', name='user_layer%d' % idx)
        itemlayer = Dense(itemlayers[idx],  activation='relu', name='item_layer%d' % idx)
        dmf_user_latent = userlayer(dmf_user_latent)
        dmf_item_latent = itemlayer(dmf_item_latent)
    dmf_vector = multiply([dmf_user_latent, dmf_item_latent])

    # MLP part
    MLP_Embedding_User = Dense(layers[0]//2, activation="linear" , name='user_embedding')
    MLP_Embedding_Item  = Dense(layers[0]//2, activation="linear" , name='item_embedding')
    mlp_user_latent = MLP_Embedding_User(user_rating)
    mlp_item_latent = MLP_Embedding_Item(item_rating)
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, mlp_num_layer):
        layer = Dense(layers[idx], activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate DMF and MLP parts
    predict_vector = concatenate([dmf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)

    model_ = Model(inputs=[user_rating, item_rating],
                   outputs=prediction)

    return model_
