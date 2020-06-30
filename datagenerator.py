import tensorflow as tf
import numpy as np
import scipy.sparse as spr

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, user, item, batch_size, negative, shuffle=True):
        self._user = user
        self._item = item
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._negative = negative
        self._total_len = item.apply(len).sum()
        self.make_one_hot_dict()
        self.make_dok_mat()
        self.make_longform()
        self.on_epoch_end()

    def make_one_hot_dict(self):
        user_dict = dict()
        item_dict = dict()
        for u, songs in zip(self._user, self._item):
            user_dict[u] = songs
            for song in songs:
                if item_dict.get(song):
                    item_dict[song].append(u)
                else:
                    item_dict[song] = [u]
        self._user_dict = user_dict
        self._item_dict = item_dict

        self._n_user = len(user_dict)
        self._n_item = len(item_dict)

    def __len__(self):
        if self._shuffle == True:
            return ((self._total_len * (self._negative+1)) // self._batch_size) + 1
        else:
            return (self._total_len // self._batch_size) + 1

    def __getitem__(self, index):
        begin = index * self._batch_size

        if index + 1 == self.__len__():
            end = self._total_len
        else:
            end = (index + 1) * self._batch_size

        indices = self._indices[begin:end]
        X,y = self.__data_generation(indices)
        return X,y

    def on_epoch_end(self):
        self._indices = np.arange(self._total_len)
        if self._shuffle == True:
            np.random.shuffle(self._indices)

    def __data_generation(self, indices):
        user_data = np.zeros((len(indices), len(self._item_dict)))
        item_data = np.zeros((len(indices), len(self._user_dict)))

        if self._shuffle == True:
            neg_user_data = np.zeros((len(indices)*self._negative, len(self._item_dict)))
            neg_item_data = np.zeros((len(indices)*self._negative, len(self._user_dict)))

        for i, ID in enumerate(indices):
            user = self._longform[ID][0]
            item = self._longform[ID][1]
            user_index = self._user_dict[user]
            item_index = self._item_dict[item]
            user_data[i][user_index] = 1
            item_data[i][item_index] = 1

        labels = np.ones(len(indices))

        if self._shuffle == True:
            c = 0
            for _, ID in enumerate(indices):
                user = self._longform[ID][0]
                for _ in range(self._negative):
                    neg_item = np.random.randint(self._n_item)
                    while self._dok_matrix.get((user,neg_item)):
                        neg_item = np.random.randint(self._n_item)
                    user_index = self._user_dict[user]
                    neg_item_index = self._item_dict[neg_item]

                    neg_user_data[c][user_index] = 0
                    neg_item_data[c][neg_item_index] = 0
                    c+=1
            user_data = np.vstack([user_data,neg_user_data])
            item_data = np.vstack([item_data,neg_item_data])
            labels = np.ones(len(indices)*(self._negative + 1))
        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((user_data,item_data),labels)
            .repeat()
            .shuffle(1025)
            .batch(self._batch_size)
            .prefetch(AUTO)
        )
        return (user_data,item_data), labels

    def make_dok_mat(self):
        item_row = np.repeat(range(len(self._user)), self._item.apply(len))
        item_col = []
        for items in self._item:
            item_col.extend(items)
        item_dat = np.repeat(1, self._total_len)

        self._dok_matrix = spr.csr_matrix((item_dat,(item_row,item_col)), shape=(len(self._user),len(self._item))).todok()

    def make_longform(self):
        self._longform = np.array(list(self._dok_matrix.keys()))
