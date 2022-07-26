import pandas as pd
import numpy as np
import re
import time
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import SpatialDropout1D, GlobalMaxPool1D
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

class LSTMwithGLOVE:
    
    
    MAX_NB_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 10
    EMBEDDING_DIM = 50
    def init(self):
        pass
    
    
    def clean_text(self,text):
    
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text) 
        text = text.replace('x', '')
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        return text
    
    
    
    def getGloveEmbeddings(self, word_index):
        
        embeddings_index = {}
        f = open('glove.6B.50d.txt', encoding='utf8')
        
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        f.close()
        embedding_matrix = np.random.random((len(word_index) + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
        
        
    def dataPreprocess(self):
        df = pd.read_csv('formspring_processed.csv')
        df = df.reset_index(drop=True)
        
        df['tweet'] = df['tweet'].apply(self.clean_text)
        df['tweet'] = df['tweet'].str.replace('\d+', '')
    
        
        
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['tweet'].values)
        word_index = tokenizer.word_index
        #print('Found %s unique tokens.' % len(word_index))
        X = tokenizer.texts_to_sequences(df['tweet'].values)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        #print('Shape of data tensor:', X.shape)
        Y = df['label'].values
        #print('Shape of label tensor:', Y.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.05, random_state = 42)

        sm = SMOTE(random_state=33)
        X_train_new, Y_train_new = sm.fit_resample(X_train, Y_train)
        '''
        undersample = RandomUnderSampler(sampling_strategy='majority')
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        X_train_new, Y_train_new = undersample.fit_resample(X_train, Y_train)
        '''
        X_train_new, Y_train_new = shuffle(X_train_new, Y_train_new)
        X_test, Y_test = shuffle(X_test, Y_test)




        return X_train_new, X_test, Y_train_new, Y_test, word_index
    
    
    
    def modelLSTM(self,optimizer='rmsprop', init='glorot_uniform'):

        X_train_new, X_test, Y_train_new, Y_test, word_index = self.dataPreprocess()
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                                            self.EMBEDDING_DIM,
                                            weights = [self.getGloveEmbeddings(word_index)],
                                            input_length = self.MAX_SEQUENCE_LENGTH,
                                            trainable=False,
                                            name = 'embeddings'))
        model.add(LSTM(60, return_sequences=True,name='lstm_layer'))
        model.add(GlobalMaxPool1D())
        model.add(Dropout(3))
        model.add(Dense(1, activation='sigmoid', init=init))


        #opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics = ['accuracy'])
        '''
        epochs = 15
        batch_size = 64
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
        accr = model.evaluate(X_test,Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        
         plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        predictedOutput = model.predict_classes(X_test, batch_size=10,verbose=0)
        print(predictedOutput)
        cm = confusion_matrix(Y_test, predictedOutput)
        print(cm)

        df_cm = pd.DataFrame(cm, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.show()
        '''
        return model
        
        
    def gridSearchCv(self,  X_train, Y_train, X_test, Y_test, embedding_matrix, word_index):
        start = time.time()
        model = KerasClassifier(build_fn=self.modelLSTM)
        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = np.array([50, 100, 150])
        batches = np.array([5, 10, 20])
        param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(X_train, Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        print("total time:", time.time() - start)

    
    
    
    
if __name__ == '__main__':
    ob = LSTMwithGLOVE()
    X_train, X_test, Y_train, Y_test, word_index = ob.dataPreprocess()
    embedding_matrix = ob.getGloveEmbeddings(word_index)
    ob.gridSearchCv(X_train, Y_train, X_test, Y_test, embedding_matrix, word_index)