from sklearn import model_selection, naive_bayes, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

def trainModel(classifier, ClassifierModelPath, vectorizer, VectorModelPath, TX, TY, VX, VY):
    classifier.fit(TX, TY)                                          #Traino il classifier
    pickle.dump(classifier, open(ClassifierModelPath, "wb"))        #Dopo averlo trainato lo salvo nel file "./ModelTFIDFVec.txt"
    pickle.dump(vectorizer, open(VectorModelPath, "wb"))            #Salvo il vectorizer nel file "./VectorModel"

    predictions = classifier.predict(VX)                            #Provo a predictare i valori di ValidX
    return metrics.accuracy_score(predictions, VY)                  #Restituisco l'accuracy


trainFile = "./train.txt"
ClassifierModelFile = "./ModelliSalvati/ClassifierModel.txt"
VectorModelFile = "./ModelliSalvati/VectorModel.model"
dataFrame = pd.read_csv(trainFile, names=['label', 'sentence'], sep=";")    #dataFrame = {labesl: [valore, valore, ...], sentence: [valore, valore, ...]}

#Leggo il file rendendolo un dizionario label-sentence. TrainX = valori dati, risultato (A o B) è in TrainY. Train si usa per trainare. Per fare il test si usano ValidX e ValidY
TrainX, ValidX, TrainY, ValidY = model_selection.train_test_split(dataFrame['sentence'].values, dataFrame['label'].values, test_size=0.2, random_state=1000)
#X -> DATI, Y -> SOLUZIONI

le = preprocessing.LabelEncoder()   #Label Encoder
TrainY = le.fit_transform(TrainY)   #Trasformo in numeri leggibili (0, 1, ...)
ValidY = le.fit_transform(ValidY)

vctr = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000) #Vectorizer
TrainX = vctr.fit_transform(TrainX) #TrainX TF IDF - Trasformo in numeri leggibili i testi
ValidX = vctr.transform(ValidX)     #ValidX TF IDF -

accuracy = trainModel(naive_bayes.MultinomialNB(), ClassifierModelFile, vctr, VectorModelFile, TrainX, TrainY, ValidX, ValidY)

print(f"Model accuracy: {accuracy*100}%")
