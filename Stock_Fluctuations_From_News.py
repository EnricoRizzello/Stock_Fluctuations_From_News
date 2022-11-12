'''
 Tentativo di predirre i cambiamenti nello stocks pricing in base ai 
 sentimenti degli utenti correlati alle news, in quanto questi sono 
 proni ad investire o meno in un'azienda in base al loro stato emotivo 
 che può alterato da certi eventi correlati all'azienda stessa.

 Esempio: 
 Se si diffonde la notizia di una grandinata che ha distrutto tutti i 
 vigneti di Tavernello è molto probabile che gli utenti nel prossimo
 anno saranno meno propensi ad investire nell'azienda e questo causerà un
 abbassamento dello stock value della ditta. 
'''
import pandas as pd

'''
 Per vettorizzare i dati useremo la funzione di CountVectorizer oppure 
 TfidVectorizer di sklearn che permette di trasformare ogni frase 
 in vettori.
 Importiamo anche RandomForestClassifier sempre da sklearn e in alternativa
 anche l'algoritmo di naive_bayes.
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
'''
 Importo da sklearn le metriche per controllare l'accuratezza dei risultati
''' 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
'''
 Per il calcolo del tempo di esecuzione
''' 
import time
'''
 Caricamento settings:
'''

print("\n--- STOCK FLUCTUATIONS FROM NEWS PREDICTIONS ---\nChoose your settings:\n")
print("|          Vectorizer |                    Classifier |")
print("| Count        0      | RandomForest           0      |")
print("| TfidF        1      | MultinomialNB          1      |\n")

settings =input()
settings = settings.replace(" ","")
print("\n")
start_time = time.time()
'''
 Caricamento del dataset 
 Source kaggle 
'''
df=pd.read_csv('DataSet.csv', encoding = "ISO-8859-1")
'''
 Nota ISO-8859-1 per gestire i caratteri speciali in lettura
 Ogni riga del dataset è composta da una data 25 frasi e una label
 associata alle frasi che può avere come valore 0 o 1
 0 => Stock price decrease nel prossimo anno
 1 => Stock price increase nel prossimo anno
 Ci sono 10 righe per ogni azienda
 I dati sono da 2002 al 2016

 Di base l'idea è di generare dalle frasi dei vettori mediante un NPL
 process e dopo aver compiuto dei passaggi di data preprocessing.
 L'idea è di creare un processo di machine learning che associ 
 determinati eventi i quali sono stati vettorizzati con un certo
 algoritmo ad una certa Label di incremento o decremento dello 
 stock price nell'anno successivo.
'''
#print(df.head())
'''
 Definiamo il training set e il test set in base alla data 
 E' importante che non sia diviso randomicamente perchè è un task
 che richiede di predirre dei valori FUTURI in base ad eventi 
 PRECEDENTI
'''
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
'''
 Nota: La colonna delle date non verrà utilizzata
 Viene rimossa tutta la punteggiatura
 Nota: tutti i caratteri diversi da una lettera maiuscola o miniscola
       vengono sostituiti da uno spazio
'''
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

'''
 Rinomino le colonne per rendere più semplice l'accesso
'''
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
#print(data.head(5))

'''
 Converto tutte le lettere da uppercase a lowercase
'''
for index in new_Index:
    data[index]=data[index].str.lower()
#print(data.head(1))


'''
 Il prossimo passo consiste di ogni riga combinare le 25 frasi per
 poter ottenere un unico testo da cui ottenere i vettori (esempio)
'''
#print(' '.join(str(x) for x in data.iloc[1,0:25]))
'''
 Quindi a partire da questo scrittura posso ottenere una lista di 
 testi unificati
'''
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#print(headlines[0])


'''
 Implementazione del Bag Of Words a partire da un algoritmo di vettorizzazione 
 settando 2 vettori con 2 n grammi

 Esempio:
    1 the tomato is red
    2 the potato is brown
    3 the tomato is shiny
    4 the potato stinks

 Sentence                 [Input Features]                [Output Feature]                 
            the tomato potato is red brown shiny stinks        Label
    1        1    1      0     1  0    0     0     0             1
    2        1    0      1     1  0    1     0     0             1
    3        1    1      0     1  0    0     1     0             0
    4        1    0      1     0  0    0     0     1             0    

 Se n grams = 2:  (the) (tomato) => (the tomato)  

'''
if(settings[0]=='0'):
    vector=CountVectorizer(ngram_range=(2,2))
else:
    vector=TfidfVectorizer(ngram_range=(2,2))
traindataset=vector.fit_transform(headlines)
'''
 Si noti che il traindataset in questo momento è una matrice sparsa
 composta da 1 e 0
'''
#print(traindataset)

'''
 Implementiamo un algoritmo di classificazione impostando il numero 
 di estimatori a 200 e il criterio di stima con entropia nel caso
 usassimo il Random Forest Classifier
 
 Nota: Si ricordino i parametri di RandomForestClassifier:
       RandomForestClassifier(bootstrap=True, 
                              class_weight=None, 
                              criterion='entropy',
                              max_depth=None, 
                              max_features='auto', 
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0, 
                              min_impurity_split=None,
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              min_weight_fraction_leaf=0.0, 
                              n_estimators=200, 
                              n_jobs=None,
                              oob_score=False, 
                              random_state=None, 
                              verbose=0,
                              warm_start=False)
'''
if(settings[1]=='0'):
    classifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
else:
    classifier=MultinomialNB()
classifier.fit(traindataset,train['Label'])

'''
 Preprocessing del test set in modo analogo al training set
'''

test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
'''
 Ottengo un test_dataset partendo dai vettori ottenuti del trainingset
 ed effettuo delle predizioni partendo dal classificatore
 allenato dal trainingset sui valori del testset
'''    
test_dataset = vector.transform(test_transform)
predictions = classifier.predict(test_dataset)


if(settings[0]=='0'):
    Vectorizer="CountVectorizer"
else:
    Vectorizer="TfidfVectorizer"

if(settings[1]=='0'):
    Classifier="RandomForestClassifier"
else:
    Classifier="MultinomialNaiveBayes"

'''
 Analisi dei risultati ottenuti verificando: 
 matrice di confusione
 accuratezza
 precision, recall f1-score dei valori 0 ed 1, di micro, macro e weighted avg  
''' 

print("--- Results using "+Vectorizer + " and " +Classifier + " ---\n")
matrix=confusion_matrix(test['Label'],predictions)
print("Confusion Matrix:")
print(matrix)
score=accuracy_score(test['Label'],predictions)
print("\nAccuracy:")
print(('%.3f'%(score))+"%")
report=classification_report(test['Label'],predictions)
print("\nReport:")
print(report)
executionTime = '%.2f'%(time.time() - start_time)
print("\nExecution time: %s seconds\n" % executionTime)







