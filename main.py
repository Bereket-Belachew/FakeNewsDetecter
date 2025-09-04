import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from wordcloud import WordCloud
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.layers import LSTM, Dropout, Dense, Embedding
from keras import Sequential

basepath = "/Users/Newstandard/PycharmProjects/MachinelearningProjects/FakeNewsDetecter/News _dataset"
true_path = os.path.join(basepath,"True.csv")
false_path= os.path.join(basepath,"Fake.csv")
df= pd.read_csv(true_path)
bf=pd.read_csv(false_path)

df['label']=1
bf['label']=0

concat_df = pd.concat([df,bf],ignore_index=True)
concat_df= shuffle(concat_df,random_state=42).reset_index(drop=True)
print(concat_df.head())

#Lets start Cleaning the Data here:

concat_df['Clean_txt'] = concat_df['Clean_txt'].str.replace('[^a-z\s]', ' ', regex=True)
concat_df['Clean_txt'] = concat_df['Clean_txt'].str.replace('\n',' ', regex=True)
concat_df['Clean_txt'] = concat_df['Clean_txt'].str.replace('\s+',' ', regex=True)


#Lets remove all the stop words with in the data
stop= stopwords.words('english')
concat_df['Clean_txt']=concat_df['Clean_txt'].apply(lambda x: " ".join([word for word in x.split() if word not in stop]))

print(concat_df['Clean_txt'][0])


#Lets plot the graph
all_words = " ".join([words for words in concat_df['Clean_txt']])
wordcloud = WordCloud(width =800, height=500, random_state=42,max_font_size=100).generate(all_words)
#This is for all the news:
plt.figure(figsize=(15,9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#What about only for the True News
true_word= " ".join([words for words in concat_df[concat_df['label']==1]['Clean_txt']])
true_word_cloud = WordCloud(width =800, height=500, random_state=42,max_font_size=100).generate(true_word)

plt.figure(figsize=(15,9))
plt.imshow(true_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#For the False news
false_word= " ".join([words for words in concat_df[concat_df['label']==0]['Clean_txt']])
false_word_cloud = WordCloud(width =800, height=500, random_state=42,max_font_size=100).generate(true_word)

plt.figure(figsize=(15,9))
plt.imshow(false_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#lets create a Word embedding
tokenizer= Tokenizer()
tokenizer.fit_on_texts(concat_df['Clean_txt'])
word_index= tokenizer.word_index
vocab_size=len(word_index)


#lets do padding

sequences = tokenizer.texts_to_sequence(df['clean_news'])
padded_seq = pad_sequence(sequences,maxlen=500,padding='post',truncating='post')


#Now lets create the embedding matrix
embedding_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word]=coefs

embedding_matrix = np.zeros((vocab_size+1,100))
for word,i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector



#lets split the data into training data and testing data
x_train,x_test,y_train,y_test= train_test_split(padded_seq,df['label'],test_size=0.20,random_state=42,stratify=['label'])

#Lets Create the Model
model = Sequential([
    Embedding(vocab_size+1,100, weights=[embedding_matrix],trainable=False),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(256),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

history = model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data= (x_test,y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()