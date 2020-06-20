from collections import Counter
import numpy as np
import pandas as pd

train_path = './data/train.json'
val_path = './data/val.json'
test_path = './data/test.json'

class Preprocess:
    def __init__(self, min_value):
        self.train = pd.read_json(train_path, typ='frame')
        self.val = pd.read_json(val_path, typ='frame')
        #self.test = pd.read_json(test_path, typ='frame')
        self.concat_train_val()
        self.min_value = min_value
        self.tag_encoder, self.tag_decoder = self.makeDict(self.total_plylst['tags'],min_value)
        self.song_encoder, self.song_decoder = self.makeDict(self.total_plylst['songs'],min_value)
        self.plylst_encoder, self.plylst_decoder = self.makeDict(self.total_plylst['id'])
        self.encoding_data()

    def encoding_data(self):
        self.total_plylst['songs'] = self.total_plylst['songs'].apply(lambda x: list(map(lambda y:self.song_encoder[y],list(filter(lambda z: z in self.song_encoder,x)))))
        self.total_plylst['tags'] = self.total_plylst['tags'].apply(lambda x: list(map(lambda y:self.tag_encoder[y],list(filter(lambda z: z in self.tag_encoder,x)))))
        self.total_plylst['id'] = self.total_plylst['id'].apply(lambda x: self.plylst_encoder[x])

    def make_longform(self, item):
        dataset = np.zeros((sum(self.total_plylst[item].apply(len)),3))
        t = 0
        for i in self.total_plylst.index:
            for item_id in self.total_plylst.iloc[i][item]:
                dataset[t] = np.array([i, item_id, 1])
                t+=1
        return dataset

    def concat_train_val(self):
        self.train['istrain'] = 1
        self.val['istrain'] = 0
        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.total_plylst = pd.concat([self.train,self.val],ignore_index=True)

    def makeDict(self,df, min_value=0):
        c = Counter()
        #if isinstance(df, pd.DataFrame):
        if isinstance(df[0], list):
            for row in df:
                c.update(row)
            c = list(filter(lambda x: x[1]>min_value, c.items()))
            enc, dec = dict(), dict()

            for i, song in enumerate(c):
                enc[song[0]] = i
                dec[i] = song[0]
            return enc, dec
        else:
            enc = dict((v,i) for i,v in enumerate(df))
            dec = dict((i,v) for i,v in enumerate(df))

            return enc, dec
