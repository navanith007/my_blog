---
layout: post
title:  "Stock Market Prediction using Artificial Intelligence(AI)"
date:   2020-12-17 12:23:12 +0530
category: Deep Learning
---

<img src="../../../../assets/images/SM_Prediction/BuySell-Stocks.jpg" width="700" height="400" />
- [1. Business Problem](#1-business-problem)
- [2. Use of Machine Learning and Deep Learning](#2-use-of-machine-learning-and-deeplearning)
- [3. Source of Data](#3-source-ofdata)
- [4.Existing Approaches](#4existing-approaches)
- [5.My Approach](#5myapproach)
- [6.Understanding the Data](#6understanding-thedata)
  - [6.1 DJIA Stock data and Reddit news](#61-djia-stock-data-and-redditnews)
  - [6.2Microsoft Stock data and technology news](#62microsoft-stock-data-and-technology-news)
- [7. Exploring the data.](#7-exploring-thedata)
  - [7.1 DJIA Stock Data and Reddit news](#71-djia-stock-data-and-redditnews)
  - [7.2 Microsoft stock data and it's technology news.](#72-microsoft-stock-data-and-its-technology-news)
- [8. Text encoding and ML models explanation](#8-text-encoding-and-ml-models-explanation)
  - [8.1 Text encoding](#81-textencoding)
  - [8.2 ML models](#82-mlmodels)
- [9. Deep Learning Model](#9-deep-learningmodel)
- [10. Future Work](#10-futurework)
- [Code Resource](#code-resource)

<hr>  

# 1. Business Problem  
Stock Market Prediction offers great profit avenues and is a fundamental stimulus for most researchers in this area.   There are two ways to predict the stock market : 1. Technical Analysis and 2: Fundamental Analysis.  

1. **Technical Analysis** uses the market price data to predict the future movements or prices of the stock.
2. **Fundamental Analysis** uses the unstructured textual information like financial news and earning reports.  
   
There are many problems can be formulated using stock market data. In this article, I'm writing about the problem which uses widely in the research area. To understand the problem you need to know the basics of stock market. 

Every company's stock market opens with an Open price and ends with a Close price in a given day. These prices are determined by the sellers and buyers of the stock.  

1. **Sellers control :** If Close price is greater than the Open price in a given day, it means Sellers are in control of the given company's stock price on that particular day.
2. **Buyers control :** If Close price is less than the Open price in a given day, it means Buyers are in control of the given company's stock price on that particular day.  
   
In this article I'm going to explain how to predict whether buyers or sellers are in control in the next day by using using both technical analysis and fundamental analysis.  

<hr>

# 2. Use of Machine Learning and Deep Learning  
In the past, researchers only used historical stock market data to predict the future stock market. They mainly depends on the time-series analysis. But, the stock data is non-stationary and very volatile which leads it difficult to use traditional time-series models. And also stock market price heavily depends on the traders sentiment, where the time-series models doesn't use the sentiments. It turns out that the traders sentiment heavily depend on the financial news articles. So, with the advancement of Machine Learning and Deep Learning and also there are so many publicly available news datasets we can use AI to predict the stock market.  

<hr>  

# 3. Source of Data
In my research, I've used two different types of datasets.  

1. Dow Jones Industrial Average (DJIA) stock market data set and the top 25 reddit headlines in the period of 2008–06–08 to 2016–07–01. You can get the data set from [here](https://www.kaggle.com/aaron7sun/stocknews).  
2. The second data set is the stock market data of a specific company (Microsoft) has been used and also the financial news articles have been used in the period of March 10th, 2014 to August 10th, 2014. News data can be find [here](https://www.kaggle.com/uciml/news-aggregator-dataset). stock data can be find [here](https://medium.com/r?url=https%3A%2F%2Ffinance.yahoo.com%2Fquote%2FMSFT%2Fhistory%3Fp%3DMSFT).  
   
<hr>  

# 4.Existing Approaches   
There are different approaches, but most of researchers either use technical analysis or fundamental analysis. For the above datasets there is a one [git hub](https://github.com/EmielStoelinga/CCMLWI) repository which explains how to use fundamental analysis. There is another [research](https://www.researchgate.net/publication/318298991_Predicting_Stock_Market_Behavior_using_Data_Mining_Technique_and_News_Sentiment_Analysis) on how to use technical analysis on the same problem but different data set.  
<hr>  

# 5.My Approach  
I've used both technical analysis and fundamental analysis on the data sets by referring to the above two research papers and able to improve the accuracy by 5% on the second data set.  
<hr>  

# 6.Understanding the Data  
Note: The below shown data is not raw data they are preprocessed data.  

## 6.1 DJIA Stock data and Reddit news  
<img src="../../../../assets/images/SM_Prediction/data_1.png"/>   
The above data is the DJIA stock data. let's understand the categorical features which are generated from the OHLC values.  

If close price of today is greater than the yesterday the move_close is increased otherwise it's decreased. Similarly for move_open, move_high, move_low are generated. move_close_open is whether close price is greater than open price on that day.  

Along with the stock data we have top 25 headlines of reddit news also.  

## 6.2Microsoft Stock data and technology news  
<img src="../../../../assets/images/SM_Prediction/data_2.png"/>     
The above data set is the stock data of Microsoft company with normalized_title is the technological related news of Microsoft published on that day. `tomorrow` is the class label which represents the move_close_price on next day.  
<hr>  

# 7. Exploring the data.  
## 7.1 DJIA Stock Data and Reddit news  
Let's see how the class label distributed.  
<img src="../../../../assets/images/SM_Prediction/class_label.png"/>    
You can clearly see that the class label is balanced in the DJIA stock dataset.  
Similarly, we can explore word cloud of Reddit news in each class.  
<img src="../../../../assets/images/SM_Prediction/news_class_1.png"/>    
<img src="../../../../assets/images/SM_Prediction/news_class_2.png"/>    

From the above word clouds we can clearly see that the news contains mostly country names and some political news which doesn't related to the stock data. From this, we can infer that these news is not highly correlated with the stock data. So that the model may not perform well with this data.  

## 7.2 Microsoft stock data and it's technology news.  
<img src="../../../../assets/images/SM_Prediction/class_label_msft.png"/>   
You can see that the Microsoft stock data class label is also balanced.  
<img src="../../../../assets/images/SM_Prediction/msft_news_class1.png"/>  
<br><br>  
<img src="../../../../assets/images/SM_Prediction/msft_news_class2.png"/>  

From the above word cloud we can infer that there are not many different words between the two classes. But, may be the sentiment of the statements might be different.  
<hr>  

# 8. Text encoding and ML models explanation
## 8.1 Text encoding  
There is text data in both data sets. The important thing is How do we encode this text data to numerical so that the ML models can run on the data set. Here, we'll see the encoding techniques.  
I've used two types encoding techniques. They are : 1) Average Word2Vec, 2)weighted word2vec with the weights being IDF(Inverse Document Frequency) values of the word.  
{% highlight python %}
embeddings_dict = {} #contains ---> "word": 50d embedded vector of the word 
#using glove.6B.50d.txt file which is downloaded from google
with open(data_path + "glove/glove.6B.50d.txt", 'r', encoding = "utf-8") as f:
    for line in f:
        values = line.split() #splitting every line
        word = values[0]      #first index contains the word
        vector = np.asarray(values[1:], "float32") #seond index to last index contains the 50d vector
        embeddings_dict[word] = vector  #adding 50d vector as value to each key
glove_words =  set(embeddings_dict.keys()) #getting all the words in the golve file
{% endhighlight %}
The above code represent how to use global vectorizer to encode each word with the dimension being 50.  
{% highlight python %}
#average Word2Vec
#compute average word2vec for each review.
x_train_avg_w2v_news = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_train_news[:]): # for each review/sentence
    vector = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += embeddings_dict[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    x_train_avg_w2v_news.append(vector)
{% endhighlight %}
The above code represents how to get the average of word2vec for a given sentence.  

{% highlight python %}
tfidf_model = TfidfVectorizer()
tfidf_model.fit(x_train_news)
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_))) #dictionary of words and corresponding idf values
tfidf_words = set(tfidf_model.get_feature_names())
x_train_tfidf_w2v_news = []; 
for sentence in tqdm(x_train_news):
    vector = np.zeros(50)
    tf_idf_weight =0;
    for word in sentence.split():
        if (word in glove_words) and (word in tfidf_words):
            vec = embeddings_dict[word]
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    x_train_tfidf_w2v_news.append(vector)
{% endhighlight %}

The above code represents how to get the weighted Word2Vec for a given sentence.  
Not only the encoding of text data we need the sentiments of the text data which is very important in stock market prediction.  

{% highlight python %}
#sentiment scores for the news data
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_scores = []
sid = SentimentIntensityAnalyzer()
for sentence in tqdm(raw_data_news['clean_news'].values):
    ss = sid.polarity_scores(sentence)
    sentiment_scores.append(ss['compound'])
{% endhighlight %}
The above code represents how to get the sentiment scores of given sentence.  

## 8.2 ML models  
I've used two ML models one is KNN and another one is SVM with different encoded data on DJIA data set. But, the results are very poor test accuracy is being in the range of 48 to 52 only. The one reason being that the news data is top 25 headlines in Reddit which are not related to stock data most of the times. I'm adding the results with different models.  

<img src="../../../../assets/images/SM_Prediction/model_comparision.png"/>   
we can clearly see that the results are not great.  
<hr>  

# 9. Deep Learning Model  
<img src="../../../../assets/images/SM_Prediction/model_plot.png">  

{% highlight python %}
max_review_length = 100 #length of text that can accept into the model
X1_train = sequence.pad_sequences(X1_train, maxlen=max_review_length)
X1_test = sequence.pad_sequences(X1_test, maxlen=max_review_length)

#Create the model

embedding_vecor_length = 32

input_1 = Input(shape=(max_review_length,))
input_2 = Input(shape=(5,))

embedding_layer = Embedding(top_words, embedding_vecor_length, input_length=max_review_length)(input_1)
conv_layer = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding_layer)
max_pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
lstm_layer = LSTM(100)(max_pool_layer)

dense_layer_1 = Dense(10, activation='relu')(input_2)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)

concat_layer = Concatenate()([lstm_layer, dense_layer_2])
output_layer = Dense(1, activation='sigmoid')(concat_layer)

model = Model(inputs = [input_1, input_2], outputs = output_layer)
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
{% endhighlight %}

The Deep Learning model use both text data and stock data. It contains two input layers.  

`Input Layer 1:` It takes the news data with maximum length of words is 100.  
`Input Layer 2:` It takes the categorized stock price data. There are 5 features.Embedding Layer: It embed the each word into 32 dimensional vector from Input Layer 1.  
`Convolution Layer:` Used a convolution with kernel size 3 on the output of embedding layer.  
`Max Pooling Layer:` Used Max pool layer with pool_size is 2 on the output of Convolution Layer.  
`LSTM Layer:` Applied LSTM layer on the output of Max Pooling Layer with the number of units are 100.  
`Concatenated layer:` The output from the Input Layer 1 which is applied with two dense layers sequentially is concatenated with the output layer of LSTM.  
`The final layer:` The Final layer contains the output of 1 unit.  

After Applying this model with 100 epochs on DJIA Stock data and Reddit news data. The **test accuracy is 0.52** only but the good thing is that precision and recall are good with this model compared to machine learning model.  
When this same model applied on Microsoft stock data and the corresponding news data the **test accuracy is 0.78**. The test accuracy of 0.78 is good in stock market prediction.  
With the same Microsoft data, Using only Fundamental analysis this git hub repository got the **test accuracy of 0.73**. But by using both both Technical Analysis and Fundamental Analysis, my model **improved the accuracy by 5%**.  

<hr>  

# 10. Future Work  
After applying different models with different data sets. It seems that the deep learning model applied on data set containing the company's specific headlines gave the good accuracy. Similarly, we can use twitter tweets about the specific company and the tweets of biggest investors of the company to increase the accuracy.  

<hr>

# Code Resource  

Github link to the complete code: [https://github.com/navanith007/Stock-Market-Prediction](https://github.com/navanith007/Stock-Market-Prediction)



