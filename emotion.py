import jieba
import numpy as np
from gensim.models import KeyedVectors
from keras import Sequential
from keras.src.layers import Embedding, Bidirectional, LSTM, Dense, Activation
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager
class Psychologist:
    def __init__(self, word_list_num):
        self.model = KeyedVectors.load_word2vec_format('chinese_word_vectors\sgns.weibo.bigram', binary=False)
        self.word_list_num = word_list_num
        self.embedding_matrix = self.embed()

    def orign(self, file_path, encode, index):
        ori = []
        df = pd.read_csv(file_path, encoding=encode)
        for comment in df[index]:
            ori.append(comment)
        return ori

    def tokenize(self, train_dev_ori):
        # 索引化
        train_dev_tokens=[]
        for text in train_dev_ori:
            # 去除所有非字母、数字、汉字
            text = re.sub(r'[^\w\u4e00-\u9fff]', '', text, flags=re.UNICODE)
            words = jieba.lcut(text)
            for i, word in enumerate(words):
                try:
                    words[i]=self.model.key_to_index[word]
                except KeyError:
                    words[i]=0
            train_dev_tokens.append(words)
        return train_dev_tokens

    def normalize(self, train_dev_tokens):
        # 标准化
        n_tokens=np.array([len(tokens) for tokens in train_dev_tokens])
        # np.sum(n_tokens < token_standard)/len(n_tokens) 输出：0.9497
        token_standard=int(np.mean(n_tokens) + 2 * np.std(n_tokens))
        train_dev_pad=pad_sequences(train_dev_tokens, maxlen=token_standard,
                                padding='pre', truncating='pre')
        # 不在embedding词表中的词
        train_dev_pad[train_dev_pad>=self.word_list_num]=0
        return token_standard, train_dev_pad


    def reverse_tokenize(self, tokens):
        text=''
        for token in tokens:
            if token != 0:
                text=text+self.model.index_to_key[token]
            else:
                text=text+''
        return text

    def embed(self, embedding_dim=300):
        embedding_matrix=np.zeros((self.word_list_num, embedding_dim))
        for i in range(self.word_list_num):
            embedding_matrix[i,:]=self.model[self.model.index_to_key[i]]
        embedding_matrix=embedding_matrix.astype('float32')
        return embedding_matrix

    def split_train_dev(self, train_dev_pad, label_ori):
        # 1:积极 0:消极
        label=[1 if label == '积极' else 0 for label in label_ori]
        train_dev_target = np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(train_dev_pad, train_dev_target, test_size=0.03, random_state=42)
        return X_train, X_test, y_train, y_test

    def bilstm(self, token_standard, X_train, X_test, y_train, y_test, embedding_dim=300):
        model=Sequential()
        # 第一层：embedding (batch_size, token_standard, embedding_dim)
        model.add(Embedding(self.word_list_num,
                            embedding_dim,
                            weights=[self.embedding_matrix],
                            input_length=token_standard,
                            trainable=False))
        # 第二层: 双向lstm (batch_size, token_standard(timesteps), 2 * units)
        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        # 第三层: lstm(batch_size, units)
        model.add(LSTM(units=16, return_sequences=False))
        # 第四层
        model.add(Dense(units=1))
        model.add(Activation('sigmoid'))

        optimizer=Adam(learning_rate=1e-3)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=5,
                            validation_data=(X_test, y_test),
                            verbose=1
                            )
        return model, history

    def visualize_train(self, history):
        font_path = '微软雅黑.ttf'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.show()

    def save_model(self, model, file_path='psy_model.keras'):
        model.save(file_path)

    def load_model(self, file_path='psy_model.keras'):
        return load_model(file_path)

    def data_process(self, train_dev_ori):
        train_dev_tokens=self.tokenize(train_dev_ori)
        token_standard, train_dev_pad=self.normalize(train_dev_tokens)
        return token_standard, train_dev_pad


if __name__=="__main__":
    psy=Psychologist(50000)
    train_dev_ori=psy.orign('all.csv','gbk','evaluation')
    label_ori=psy.orign('all.csv','gbk','label')
    token_standard, train_dev_pad=psy.data_process(train_dev_ori)
    X_train, X_test, y_train, y_test =psy.split_train_dev(train_dev_pad, label_ori)
    model, history=psy.bilstm(token_standard, X_train, X_test, y_train, y_test)
    # psy.visualize_train(history)
    psy.save_model(model)




