# Sentiment-Analysis-Emotions-Pridictibility-of-Stock-Market
# Abstract
Today's market globally and locally in Pakistan Stock market prediction is incredibly vital within 
the planning of business activities. Stock worth prediction has attracted several researchers in 
multiple disciplines as well as computer science, statistics, economics, finance, and operations 
research. Recent studies have shown that the bulk quantity of online data within the property right 
like Wikipedia usage pattern, news stories from the thought media, and social media discussions 
will have gained noticeable results of investors' opinions towards money markets. The 
responsibility of the procedure models on exchange prediction is important because it is incredibly 
sensitive to the economy and might directly lead to loss. In this project named “Sentiment 
Analysis, Emotions and Predictability for Stock Market”, we have a tendency to retrieve, extract, 
and analyze the consequences of stories' sentiments on the exchange. Our main contributions 
embody the event of a sentiment analysis and opinion mining lexicon for the money sector, the 
event of a dictionary-based sentiment analysis model along with machine learning algorithms, and 
therefore the analysis of the model for gauging the consequences of stories sentiments on stocks 
or the crypto currencies
# Introduction
All Stock Prediction is a challenging problem within the field of finance similarly as engineering, 
computer science and arithmetic. Due to its gain, it's attracted a lot of attention each 
from educational facet and business facet. Stock worth prediction has continually been a subject 
matter of interest for many investors and monetary analysts.
In last ten years, stock prices or crypto currencies popularity have experienced tremendous global 
growth resulting in much higher trading prices more over this might be because of the reaction of 
the people over recent years on social media, news and RSS live feeds. Money Market movement 
is explained as up and down of the markets. The upward shift represents positive returns, while 
the downward shift represents negative returns. In last couple of days Elon Musk, the founder and 
CEO of well know Automobile company and space research firm only tweets about a crypto 
currency on social media, only a minute later the trading price of that crypto currency reached at 
its peak. The distinctive nature of social media website makes it a valuable source for mining 
public views or emotions to predict the nature of money market. Recently a huge number of 
progress has been made to study emotions using social media but there is a limited research or 
platform available for the users where they could view the prediction and the investment on stocks 
and crypto currencies along with their market caps.
Most popular task in natural language processing area is sentiment classification which predicts 
sentiment’s opinion or emotion from a given corpus. This proposed project to predict the market 
with respect to the previous result of actual market along with the sentiment and emotions of the 
people helps the investors how, when and where to invest resulting in growth and stability of the 
economy in future. 
We have calculated the MSE for all the models and then compare all the models by MSE using 
the formula <br />
![RMSE ](https://github.com/YMinhaj/Sentiment-Analysis-Emotions-Pridictibility-of-Stock-Market/blob/main/DocumentImage/rmse.png?raw=true "Round Mean Square Error Formula")<br />
However, result show result show the MSE all Machine Learning algorithms between 6-14. On 
average we have predicted the company stock for the next opening day and also predicted the 
stocks based on twitter news for a week long.
# Background and Related Work
In last ten years, stock prices or crypto currencies popularity have experienced tremendous global growth resulting in much higher trading prices more over this might be because of the reaction of the people over recent years on social media, news and RSS live feeds [1]. Money Market movement is explained as up and down of the markets. The upward shift represents positive returns, while the downward shift represents negative returns. In last couple of days Elon Musk, the founder and CEO of well know Automobile company and space research firm only tweets about a crypto currency on social media, only a minute later the trading price of that crypto currency reached at its peak [4]. The distinctive nature of social media website makes it a valuable source for mining public views or emotions to predict the nature of money market. Recently a huge number of progress has been made to study emotions using social media but there is a limited research or platform available for the users where they could view the prediction and the investment on stocks and crypto currencies along with their market caps [5]. Most popular task in natural language processing area is sentiment classification which predicts sentiment’s opinion or emotion from a given corpus. This proposed project to predict the market with respect to the previous result of actual market along with the sentiment and emotions of the people helps the investors how, when and where to invest resulting in growth and stability of the economy in future [6].
There is a lot of work done for predicting the shifts in money market. Recently, a ton of attention-grabbing work has been done in the space of applying Machine Learning Algorithms for analyzing price patterns and predicting of money market. Most investors nowadays depend on Intelligent Trading Systems which help them in predicting prices based on various situations and conditions. Multiple researches take place which focusses on some point like Chinese sentiment analysis method which predicts the only stock market price in china [3]. Now a day’s crypto currencies have done major outbreak in market no one know when the crypto currencies will shift, some of the financial researchers had predicted the machine learning model to predict the prices of stocks and evaluating them using accuracy recall and F-1 score [2].
Zhu, M. at el. [10] focused on using a naïve bayes classifier and a linear regression model to predict the opening of stock prices for ten different companies and achieved a result of an accuracy of 52.2%. Further they concluded that the relationship between the public sentiment and stocks market movements.
Shah et al. [11] shows how current sentiment can be utilised to forecast the pharmaceutical market's changes. To forecast the movements, the authors of this paper utilised a dictionary-based sentiment analysis model that exclusively used sentiment from news. On the other hand, for our research, we focused on public sentiment, scraping for tweets about various companies throughout the previous n years. In addition, we employed a linear regression model to assess the model's performance in correctly predicting stock prices. 

# Methodology
In this project we are focusing about the prediction and technical analysis of the market.  Stock-money market forecasting is the method to determine the future value of company stock. Nowadays, a huge amount of valuable information related to the financial market is available on various media such as websites, twitter, Facebook, blogs and such others. In general, a stock price depends on two factors. One is fundamental factor and another one is technical factor. The fundamental factor mainly depends on the statistical data of a company. It includes reports, financial status of the company, the balance sheets, dividends and policies of the companies whose stock are to be observed. The technical factor includes the quantitative parameters like trend indicators, daily ups and downs, highest and lowest values of a day, volume of stock, indices, put/call ratios, etc. In technical factor the historical prices are considered for the forecasting. Initially the historical prices of the selected company are downloaded from the website. Various methods of stock level indicators are available to computing the stock value. Few of them are Moving Average, Stochastic RSI (Relative-Strength Index), Bollinger bands, Accumulation – Distribution, Typical Point (pivot point).
<br />
![Block Diagram ](https://github.com/YMinhaj/Sentiment-Analysis-Emotions-Pridictibility-of-Stock-Market/blob/main/DocumentImage/blockdiagram.png?raw=true "Block Diagram")<br />
Mainly the data collected from YAHOO and Crypto exchanges needs to be preprocessed to make it suitable. But there is a main problem that there is no data when the market is closed, so to overcome this problem we use simple formula Y = (xPrevious + xNext)/2 We are using two metrices which are useful for machine learning algorithms HLPCT (High Low Percentage) (HLPCT = High-low/low) PCTChange(Percentage Change) ( PCTChange = Close-open/open ).
The data collected from crypto exchange on some interval of time will be processed and the polarity and subjectivity will be calculated accordingly. Data Collected from Twitter needs to be preprocessed to make it suitable User Request the API to get Tweets from the Server. We use Twitter Api which is a REST Api the result will come in JSON format The Search API allows filtering based on language, region, geolocation and time. JSON objects that contain the tweets and their metadata. A variety of information, including username, time, location, retweets, and more. we use $ sign as a ticker to gather the most financial tweets. used TweetPy which is a wrapper for the Twitter API.
## Tools & Platform
By analyzing the data obtained and compare the results obtained from that analysis respectively. In this project we use multiple Machine Learning Algorithms namely, Nave Bayes, Linear Regression, LSTM and ARIMA. Furthermore, after the results, we will evaluate these model’s performance using Precision, Recall and F-1 score. CV technique.
We will use the following tools and platform
I.	Visual Studio
II.	Python
III.	Live feed API
IV.	Html 5
V.	Jquery
VI.	Bootstrap

## Project Timeline
Task	TimeLine
Research Analysis
	1/Mar/2021 – 31/Mar/2021
Data gathering Methods
	1/Apr/2021 – 04/May/2021
Money Market Data
	5/May/2021 – 01/June/2021
Preparing of Data
	01/June/2021 – 31/July/2021
Environment Development
	05/May/2021 – 31/July/2021
Executing Data Models
	01/Aug/2021 – 15/Sep/2021
Analyzing Data with Model
	01/Sep/2021 – 30/Sep/2021
Results/Outcomes
	01/Oct/2021 – 31/Oct/2021
<br />
![TimeLine ](https://github.com/YMinhaj/Sentiment-Analysis-Emotions-Pridictibility-of-Stock-Market/blob/main/DocumentImage/timeline.png?raw=true "TimeLines")<br />
## Software Requirement Specification
### Introduction
Today's market globally and locally in Pakistan Stock market prediction is incredibly vital within the planning of business activities. Stock worth prediction has attracted several researchers in multiple disciplines as well as computer science, statistics, economics, finance, and operations research. Recent studies have shown that the bulk quantity of online data within the property right like Wikipedia usage pattern, news stories from the thought media, and social media discussions will have gained noticeable results of investors' opinions towards money markets. The responsibility of the procedure models on exchange prediction is important because it is incredibly sensitive to the economy and might directly lead to loss. In this project named “Sentiment Analysis, Emotions and Predictability for Stock Market”, we tend to retrieve, extract, and analyze the consequences of stories' sentiments on the exchange. Our main contributions embody the event of a sentiment analysis and opinion mining lexicon for the money sector, the event of a dictionary-based sentiment analysis model along with machine learning algorithms, and therefore the analysis of the model for gauging the consequences of stories sentiments on stocks or the crypto currencies.
### Scope
Our goal with this project is to is to provide a knowledge-intensive and computationally efficient coarse-grained analysis of Cryptocurrencies stock market values which can be analyzed by analyzing the social media comments and tweets related the stock market or crypto market date and to predict that how can it create the massive impact on the stock market as well as on the crypto market. As everyone knows that cryptocurrencies are becoming increasingly relevant in the financial world and can be considered as an emerging market. The high data availability of this market and very low barrier of entry, makes this an excellent subject that now adays people are very much interested in, particularly from social networks. This data can presumably be used to infer future human behavior, and therefore could be used to develop advantageous trading strategies [1,2] as has been shown in recent attempts to detect speculative bubbles in the cryptocurrency market using sentiment analysis [3]. Sentiment analysis has found widespread use in combination with social media, as social media is a good source of valuable and sentimental, however unstructured data itself is of little value for real world applications (IBM, 2017) and social media posts fall into this category. Therefore, sentiment analysis is the ideal tool to transform this unstructured data into tangible and processable information. 
### Problem Definition
Stock market attracts thousands of investors’ hearts from all around the world. The risk and profit of it has great charm and every investor wants to book profit from that. People use various methods to predict market volatility, such as K-line diagram analysis method, Point Data Diagram, Moving Average Convergence Divergence, even coin tossing, fortune telling, and so on. Now, all the financial data is stored digitally and is easily accessible. Availability of this huge amount of financial data in digital media creates appropriate conditions for a data mining research. The important problem in this area is to make effective use of the available data. [4]
### Social Media Sentiment Analysis