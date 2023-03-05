import pandas as pd
from datetime import datetime
from datetime import timedelta
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

Tweets_Final_Data = pd.DataFrame(
    columns=['Date', 'mean_tweets_positivity', 'mean_tweets_neutrality', 'mean_tweets_negativity',
             'mean_tweets_compound',
             'positive_tweet_count', 'positive_tweet_popularity', 'positive_verified_account_count',
             'negative_tweet_count', 'negative_tweet_popularity', 'negative_verified_account_count',
             'neutral_tweet_count', 'neutral_tweet_popularity', 'neutral_verified_account_count'])


def boolean_converter(dfcolumn):
    boolean_list = []
    for boolean in dfcolumn:
        if (boolean == True):
            boolean_list.append(1)
        elif (boolean == False):
            boolean_list.append(0)
    dfcolumn = boolean_list
    return dfcolumn


def date_converter(dfcolumn):
    date_list = []
    dfcolumn = dfcolumn.astype('datetime64')
    for date in dfcolumn:
        date = str(date.date()) + (' ') + str(date.hour) + (':00:00')
        new_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        date_list.append(new_date)
    dfcolumn = date_list
    return dfcolumn


def user_age_calculator(dfcolumn):
    date = datetime.now()
    dfcolumn = dfcolumn.astype('datetime64')
    age_list = []
    for created_date in dfcolumn:
        age = ((date.year - created_date.year) * 365) + (created_date.month * 12) + (created_date.day)
        age_list.append(age)
    return age_list


def cleanTwt(twt):
    twt = re.sub('#bitcoin', 'bitcoin', twt)  # removes the # from bitcoin
    twt = re.sub('#Bitcoin', 'Bitcoin', twt)  # removes the # from bitcoin
    twt = re.sub('#[A-Za-z0-9]+', '', twt)  # removes any strings with a '#'
    twt = re.sub('\\n', '', twt)  # removing the '\n' string
    twt = re.sub('https?:\/\/\S+', '', twt)  # removes any hyperlinks
    twt = re.sub('@[A-Za-z0-9]+', '', twt)  # removes any mentions
    return twt


def twPopularity(rt, fav, reply):
    return rt + fav + reply


def sentiment_Vader_DESC(text):
    over_all_polarity = SentimentIntensityAnalyzer().polarity_scores(text)
    return over_all_polarity


def grouped_compound(comp):
    if comp >= 0.05:
        return "positive"
    elif comp <= -0.05:
        return "negative"
    else:
        return "neutral"


sample = 12
while sample < 24:
    start_date = datetime.now()
    data = pd.read_csv('twitter_data_samples\Sample_{}.csv'.format(sample))
    data = data.drop(data.columns[0], axis=1)
    data.Date = date_converter(data['Date'])
    data.Date = data.Date.astype('datetime64')
    data.isVerified = boolean_converter(data.isVerified)
    data['user_age_by_day'] = user_age_calculator(data.user_Created_Date)
    data['cleaned_Tweet'] = data['Tweet'].apply(cleanTwt)
    data['Tweet_Popularity'] = twPopularity(data['retweetCount'], data['likeCount'], data['replyCount'])
    lists = data['cleaned_Tweet'].apply(lambda x: sentiment_Vader_DESC(x))
    end_date = datetime.now()
    print("Sample_{}.csv bitti,Geçen Süre:{}".format(sample, end_date - start_date))

    Negativity = []
    Neutral = []
    Positivity = []
    Compound = []
    Sentiment_conc = []
    count = 0
    while count < len(lists):
        neg, neut, post, comp = lists[count].values()
        Negativity.append(neg)
        Neutral.append(neut)
        Positivity.append(post)
        Compound.append(comp)
        if comp >= 0.05:
            Sentiment_conc.append("positive")
        elif comp <= -0.05:
            Sentiment_conc.append("negative")
        else:
            Sentiment_conc.append("neutral")
        count += 1
    data['Negativity'] = Negativity
    data['Neutral'] = Neutral
    data['Positivity'] = Positivity
    data['Compound'] = Compound
    data['Sentiment_Conc'] = Sentiment_conc

    verified_compounds = []
    count = 0
    while count < len(data.isVerified):
        if (data.isVerified[count] == 1):
            verified_compounds.append(data.Compound[count])
        else:
            verified_compounds.append(0)
        count += 1
    data['Verified_Compounds'] = verified_compounds

    data.to_csv('twitter_data_sentimented_samples\Sentimentend_Sample_{}.csv'.format(sample))

    data_group = data.groupby('Date') \
        .agg(mean_tweets_positivity=('Positivity', 'mean'), \
             mean_tweets_neutrality=('Neutral', 'mean'),
             mean_tweets_negativity=('Negativity', 'mean'),
             mean_tweets_compound=('Compound', 'mean'),
             mean_verified_compound=('Verified_Compounds', 'mean')
             )

    positive_group = data.query("Sentiment_Conc=='positive'").groupby('Date').agg(
        positive_tweet_count=('Tweet', 'count'),
        positive_tweet_popularity=('Tweet_Popularity', 'mean'),
        positive_verified_account_count=('isVerified', 'sum'))

    negative_group = data.query("Sentiment_Conc=='negative'").groupby('Date').agg(
        negative_tweet_count=('Tweet', 'count'),
        negative_tweet_popularity=('Tweet_Popularity', 'mean'),
        negative_verified_account_count=('isVerified', 'sum'))

    neutral_group = data.query("Sentiment_Conc=='neutral'").groupby('Date').agg(neutral_tweet_count=('Tweet', 'count'),
                                                                                neutral_tweet_popularity=(
                                                                                'Tweet_Popularity', 'mean'),
                                                                                neutral_verified_account_count=(
                                                                                'isVerified', 'sum'))
    data_group_all = pd.concat([data_group, positive_group, negative_group, neutral_group], axis=1).reset_index()

    Tweets_Final_Data = pd.concat([Tweets_Final_Data, data_group_all])
    Tweets_Final_Data = Tweets_Final_Data.reset_index()
    Tweets_Final_Data = Tweets_Final_Data.drop('index', axis=1)
    sample += 1

# Tweets_Final_Data.to_csv('Final_Tweets_Grouped\Tweets_Final_Data_2.csv')