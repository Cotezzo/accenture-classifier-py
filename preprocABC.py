import os
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Funzioni       ########################################################################################################
def preProcessFilesABC(pathIn, pathOut):
    stopWords = set(stopwords.words('italian'))
    st = SnowballStemmer('italian')
    for directory in os.listdir(pathIn):
        for file in os.listdir(f"{pathIn}/{directory}"):
            with open(f"{pathIn}/{directory}/{file}", encoding="latin1") as fIn, open(f"{pathOut}/{directory}/{file}", "w") as fOut:
                for word in re.split('[^a-z]+', fIn.read().lower()):
                    if word not in stopWords:
                        fOut.write(st.stem(word) + ' ')

#Inizio main    ########################################################################################################
preProcessFilesABC("./Multiruba/NonProcessati/", "./Multiruba/Processati/")
print("Operazione eseguita con successo. ")

