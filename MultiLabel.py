from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
import sklearn.datasets as skds
from pathlib import Path
import pickle


path_train = "./Multiruba/Processati"
FileModel = "./Multiruba/FileModel.model"
FileTokenizer = "./Multiruba/Tokenizer.model"
epochs = 10
VocabSize = 15000   #Dimensione vocabolario (numero parole diverse da memorizzare)
BatchSize = 100


files = skds.load_files(path_train, load_content=False)
train_size = int(len(files.target) * .8)
#Creo X(Dati) e Y(Soluzioni)
x_all = [Path(f).read_text(encoding="latin1") for i, f in enumerate(files.filenames)]   #Contenuto dei file
y_all = to_categorical(files.target)                                                    #Label [0, 0, 0] dei file

x_train = x_all[:train_size]                                                            #I primi contenuti x train
x_test = x_all[train_size:]                                                             #Gli ultimi contenuti x test

tokenizer = Tokenizer(num_words=VocabSize)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')                              #Tokenizzo testo
x_test = tokenizer.texts_to_matrix(x_test, mode='tfidf')                                # //

y_train = y_all[:train_size]                                                            #Le prime label x train
y_test = y_all[train_size:]                                                             #Le ultime label x test
###############################################################################

with open(FileTokenizer, 'wb') as handle:                                               #Salvo il tokenizer
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


model = Sequential()

model.add(Dense(512, input_shape=(VocabSize,), name="Layer1"))                          #Creo "strato"
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512, name="Layer2"))                                                    #Creo "strato"
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(len(files.target_names), name="OutLayer"))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=BatchSize, epochs=epochs, verbose=0)#, validation_split=0.1)

model.save(FileModel, save_format='h5')                                                 #Salvo il modello

score = model.evaluate(x_test, y_test, batch_size=BatchSize, verbose=1)
print(f'Accuracy: {round(score[1]*100, 2)}%')
