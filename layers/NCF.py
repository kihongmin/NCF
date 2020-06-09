import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Embedding,Flatten,Concatenate, Dense, Layer

class GMF_layer(Layer):
    def __init__(self,n_plylst,n_items,embed_dim,name='GMF_layer',**kwargs):
        self.mf_embedding_plylst = Embedding(input_dim=n_plylst, output_dim = embed_dim,name='mf_plylst_embedding')
        self.mf_embedding_item = Embedding(input_dim=n_items, output_dim = embed_dim,name='mf_item_embedding')
        super(GMF_layer, self).__init__(name=name, **kwargs)

    def call(self,inputs):
        input_plylst, input_item = inputs
        plylst = self.mf_embedding_plylst(input_plylst)
        item = self.mf_embedding_plylst(input_item)
        return tf.math.multiply(plylst, item)

class MLP_layer(Layer):
    def __init__(self, n_plylst,n_items,embed_dim,dims,name='MLP_layer',**kwargs):
        self.mlp_embedding_plylst = Embedding(input_dim=n_plylst, output_dim = embed_dim,name='mlp_plylst_embedding')
        self.mlp_embedding_item = Embedding(input_dim=n_items, output_dim = embed_dim,name='mlp_item_embedding')
        self.layers = []
        for dim in dims:
            self.layers.append(Dense(dim,activation='relu',name='dense_%d'%(dim)))
        super(MLP_layer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        input_plylst, input_item = inputs
        plylst = self.mlp_embedding_plylst(input_plylst)
        item = self.mlp_embedding_item(input_item)
        x = Concatenate()([plylst,item])
        for layer in self.layers:
            x = layer(x)
        return x

class NeuMF_layer(Model):
    def __init__(self, n_plylst,n_items,embed_dim, dims,name='NeuMF_layer',**kwargs):
        super(NeuMF_layer, self).__init__(name=name, **kwargs)
        self.GMF = GMF_layer(n_plylst,n_items,embed_dim)
        self.MLP = MLP_layer(n_plylst,n_items,embed_dim, dims)
        self.last_Dense = Dense(1,activation='sigmoid',name='output_layer')

    def call(self,inputs):
        gmf = self.GMF(inputs)
        mlp = self.MLP(inputs)
        concat = Concatenate()([gmf,mlp])
        output = self.last_Dense(concat)
        return output

if __name__ == '__main__':
    plylst_input = Input(shape=(1,),name='plylst_input')
    song_input = Input(shape=(1,),name='song_input')

    embed_dim, dims = 300, [64,32,16,8]

    NCF = NeuMF_layer(n_plylst,n_songs,embed_dim,dims)
    output = NCF((plylst_input,song_input))
