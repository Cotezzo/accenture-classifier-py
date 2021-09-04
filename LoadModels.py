from nltk import SnowballStemmer
from nltk.corpus import stopwords
import pickle
import re
import os

#Funzione per leggere tutti i txt di una directory
def readDirFiles(path):
    ret = []
    for Filename in os.listdir(path):                               #Per ogni file contenuto nella cartella
        with open(f"{path}{Filename}", encoding="latin1") as f:     #Lo apro
            ret.append(f.read())                                    #E ne aggiungo il contenuto all'array
    return ret

#Funzione per preprocessare tutti i txt di una directory
def process(path):
    processedFilesMatrix = []                                       #Array conenente array di parole post-processing (inizializzazione)

    stopWords = set(stopwords.words('italian'))                     #stopWords da rimuovere dal testo
    stopWords.add('')
    st = SnowballStemmer('italian')                                 #Il tipo di stemmer utilizzato per troncare le parole

    dataArr = readDirFiles(path)                                    #Chiamo la funzione per ottenere l'array conenente i file come stringa
    for data in dataArr:                                            #Per ogni elemento nell'array
        words = re.split('[^a-z0-9]+', data.lower())                #Lo rendiamo leggibile rimuovendo punteggiatura e mettendolo in lowercase
        edibleWords = ''                                            #Inizializziamo l'array contenente le parole utilizzabili
        for word in words:                                          #Per ogni parola
            if word not in stopWords:                               #Se non è una stopword
                edibleWords += st.stem(word)+" "                    #Ne aggiungo lo stem alla stringa di parole processabili
        processedFilesMatrix.append(edibleWords[:-1])               #Aggiungo le parole utilizzabili all'array sotto forma di stringa, rimuovendo lo spazio alla fine
    return processedFilesMatrix

#I percorsi delle risorse utilizzate
probaFolder = "./class/"
ClassifierModelPath = "./models/classifier.txt"
VectorModelPath = "./models/vector.model"

#Carico i modelli - classifier e vectorizer creati durante il training
classifier = pickle.load(open(ClassifierModelPath, "rb"))
vectorizer = pickle.load(open(VectorModelPath, "rb"))

#Processa i file.txt e li inserisce in un array, dopodichè li trasforma col vectorizer per renderli processabili.
X = process(probaFolder)
X = vectorizer.transform(X)

#Calcola, per ogni documento, la probabilità che sia un articolo di Sport o di qualcos'altro, e stampa il risultato.
prob = classifier.predict_proba(X)
for index, file in enumerate(os.listdir(probaFolder)):
    print(f"{file}) Sport:{round(prob[index][0]*100, 2)}%   Altro:{round(prob[index][1]*100, 2)}%")


"""
from pathlib import Path

def readDirFiles(path):
    ret = []
    for Filename in os.listdir(path):
        with open(f"{path}{Filename}", encoding="latin1") as f:
            ret.append(f.read())
    return ret
    
X = [" ".join([st.stem(w) for w in re.split('[^a-z0-9]+', data.lower()) if w not in stopWords]) for data in readFiles(trainFolder)]

#X = [" ".join([st.stem(w) for w in re.split('[^a-z0-9]+', open(f"{trainFolder}/{file}", encoding="latin1").read().lower()) if w not in stopWords]) for file in os.listdir(trainFolder)]
#X = [" ".join([st.stem(w) for w in re.split('[^a-z0-9]+', Path(f"{probaFolder}/{file}").read_text(encoding="latin1").lower()) if w not in stopWords]) for file in os.listdir(trainFolder)]
"""