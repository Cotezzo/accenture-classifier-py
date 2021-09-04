import os
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Funzioni       ########################################################################################################
def readDirFiles(path):     #Dato percorso dir, legge tutti file e restituisce array contenente questi ultimi (come stringhe)
    return [open(path + Filename, encoding="latin1").read() for Filename in os.listdir(path)]

def createPreprocessedTrainFile(directories, pathOUT):
    f = open(f"{pathOUT}train.txt", "w")
    for directory in directories:
        for data in readDirFiles(directory):
            f.write(directory[-2] + ';')
            for w in re.split('[^a-z]+', data.lower()):
                if w not in stopWords:
                    f.write(st.stem(w) + ' ')
            f.write('\n')
    f.close()

#Inizio main    ########################################################################################################
stopWords = set(stopwords.words('italian'))
st = SnowballStemmer('italian')
createPreprocessedTrainFile(["./NonProcessati/A/", "./NonProcessati/B/"], "./")
print("Operazione eseguita con successo. ")








"""
def readFile(path):         #Dato percorso di un file, ne restituisce il contenuto come stringa
    return open(path, encoding="latin1").read()

def createFiles(pathIN, pathOUT):
    for i, data in enumerate(readDirFiles(pathIN)):
        f = open(pathOUT+"{}.txt".format(i), "w")
        for w in re.split('[^a-z]+', data.lower()):
            if w not in stopWords:
                f.write(st.stem(w) + ' ')
        f.close()
"""

"""
def readFiles(path1, path2):
    txts = []
    for File1, File2 in zip(os.listdir(path1), os.listdir(path2)):
        txts.append(open(path1+File1, encoding="utf8").read())
        txts.append(open(path2+File2, encoding="utf8").read())
    return txts
"""

"""
dataArr = readDirFiles("./Sport/")                          #Leggo tutti i file e li metto in un array
for i, data in enumerate(dataArr):                      #Per ogni stringa (assieme al contatore)
    f = open("./Preprocessati/{}.txt".format(i), "w")   #Sovrascrivo/Creo un file 
    words = re.split('[^a-z0-9]+', data.lower())        #Separo tutte le parole rimuovendo la punteggiatura e mettendo in minuscolo
    for w in words:                                         #Per ogni parola
        if w not in stopWords:                              #Se non è contenuta in stopWords
            f.write(st.stem(w)+' ')                         #La scrivo nel file appena creato con ' ' per separare le parole
    f.close()                                               #Che poi chiudo

import nltk
nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import ItalianStemmer
"""

"""
#lem = ItalianStemmer()
#lem = WordNetLemmatizer()

processedFilesMatrix = [[st.stem(w) for w in re.split('[^a-z0-9]+', data.lower()) if w not in stopWords] for data in readFile("./Sport/")] #Tutta la roba sotto collassata su una linea
print(processedFileMatrix)

Si traduce in

processedFilesMatrix = []                           #Array conenente array di parole post-processing (inizializzazione)
dataArr = readFile("./Sport/")                      #Chiamo la funzione per ottenere l'array conenente i file come stringa
for data in dataArr:                                    #Per ogni elemento nell'array
    words = re.split('[^a-z0-9]+', data.lower())        #Lo rendiamo leggibile rimuovendo punteggiatura e mettendolo in lowercase
    wordsEdible = []                                    #Inizializziamo l'array contenente le parole utilizzabili
    for w in words:                                         #Per ogni parola
        if w not in stopWords:                              #Se non è una stopword
            wordsEdible.append(lem.stem(w))                  #Ne aggiungo lo stem all'array di parole commestibili
    processedFilesMatrix.append(wordsEdible)            #Aggiungo le parole utilizzabili all'array (matrice 2d)
    
    
processedFilesMatrix = [st.stem(w) for w in re.split('[^a-z0-9]+', open(PATH, encoding="latin1").read().lower()) if w not in stopWords] 

words = re.split('[^a-z0-9]+', open(PATH, encoding="latin1").read().lower())        
wordsEdible = []                                    
for w in words:                                       
    if w not in stopWords:                              
        wordsEdible.append(lem.stem(w))                  
"""
