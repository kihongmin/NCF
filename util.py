from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as spr

train_path = './data/train.json'
val_path = './data/val.json'
test_path = './data/test.json'

class Preprocess:
    def __init__(self, min_value,negative):
        self.train = pd.read_json(train_path, typ='frame')
        self.val = pd.read_json(val_path, typ='frame')
        #self.test = pd.read_json(test_path, typ='frame')
        self.concat_train_val()
        self.min_value = min_value
        self.tag_encoder, self.tag_decoder = self.makeDict(self.total_plylst['tags'],min_value)
        self.song_encoder, self.song_decoder = self.makeDict(self.total_plylst['songs'],min_value)
        self.plylst_encoder, self.plylst_decoder = self.makeDict(self.total_plylst['id'])

        self.n_tag = len(self.tag_encoder)
        self.n_song = len(self.song_encoder)
        self.n_plylst = len(self.plylst_encoder)
        self.negative = negative

        self.encoding_data()

    def run(self, item_name):
        self.make_mat(item_name)
        return self.get_train_instances()

    def encoding_data(self):
        self.total_plylst['songs'] = self.total_plylst['songs'].apply(lambda x: list(map(lambda y:self.song_encoder[y],list(filter(lambda z: z in self.song_encoder,x)))))
        self.total_plylst['tags'] = self.total_plylst['tags'].apply(lambda x: list(map(lambda y:self.tag_encoder[y],list(filter(lambda z: z in self.tag_encoder,x)))))
        self.total_plylst['id'] = self.total_plylst['id'].apply(lambda x: self.plylst_encoder[x])

    def make_mat(self,item_name):
        item_row = np.repeat(range(len(self.total_plylst)), self.total_plylst[item_name].apply(len))
        item_col = []
        for items in self.total_plylst[item_name]:
            item_col.extend(items)
        item_dat = np.repeat(1, self.total_plylst[item_name].apply(len).sum())

        if item_name == 'songs':
            self.spr_songs = spr.csr_matrix((item_dat,(item_row,item_col)), shape=(self.n_plylst,self.n_song)).todok()
        elif item_name == 'tags':
            self.spr_tags = spr.csr_matrix((item_dat,(item_row,item_col)), shape=(self.n_plylst,self.n_song)).todok()

    def get_train_instances(self,negative=True):
        user_input, item_input, labels = [], [], []
        train = self.spr_songs
        num_users = train.shape[0]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            if negative == True:
                for t in range(self.negative):
                    j = np.random.randint(self.n_song)
                    while (u, j) in train.keys():
                        j = np.random.randint(self.n_song)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
        return user_input, item_input, labels

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
