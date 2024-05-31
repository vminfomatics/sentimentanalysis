
import streamlit as st
import numpy as np
import pandas as pd
import json
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import time


# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import requests
#sns.set_style('darkgrid')



df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet"])
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

        
def get_tweets_from_file():
    df = pd.read_csv("data.csv")
    df = df.sample(n=100)
    df_sample = df[['Date', 'User', 'IsVerified', 'Tweet']]
    return df_sample
    
    # Function to Clean the Tweet.
def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def prepCloud(Topic_text,Topic):
    Topic = str(Topic).lower()
    Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
    Topic = re.split("\s+",str(Topic))
    stopwords = set(STOPWORDS)
    stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
    ###
    text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
    return text_new
    
def analysis(Topic):
    # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
    with st.spinner("Please wait, Data are being extracted"):
        #get_tweets(Topic , Count=500)
        df = get_tweets_from_file()
    
    st.write(df)
    st.success('Data from Social Site have been Extracted !!!!')    
    df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
    df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
    st.write("Total Data Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
    st.write("Total Positive Reviews are : {} | Total Negative Reviews are : {} | Total Neutral Reviews are : #{}".format(len(df[df["Sentiment"]=="Positive"]),len(df[df["Sentiment"]=="Negative"]), len(df[df["Sentiment"]=="Neutral"])))
    st.write(df[["Date","User","IsVerified","Tweet",'clean_tweet']])
    st.subheader(" Count Plot for Different Sentiments")
    st.write(sns.countplot(df["Sentiment"]))
    st.pyplot(plt.gcf())
    
    st.subheader(" WordCloud for Positive Tweets")
    text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
    stopwords = set(STOPWORDS)
    text_new_positive = prepCloud(text_positive,Topic)
    #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
    wordcloud = WordCloud(stopwords=stopwords,max_words=100,max_font_size=70).generate(text_new_positive)
    st.write(plt.imshow(wordcloud, interpolation='bilinear'))
    st.pyplot(plt.gcf())
    
    st.subheader(" WordCloud for Negative Tweets")
    text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
    stopwords = set(STOPWORDS)
    text_new_negative = prepCloud(text_negative,Topic)
    #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
    wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
    st.write(plt.imshow(wordcloud, interpolation='bilinear'))
    st.pyplot(plt.gcf())

def main():

    
    st.sidebar.subheader("GES's R. H. Sapat College of Engineering, Management Studies and Research, Nashik")
    st.subheader("Prediction of Indian election using sentiment analysis on Twitter (X) data")
    
 
    menu = ["Home","Login","SignUp","Text Analysis", "Analysis File"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    

    if choice == "Home":
        text = """The project Prediction of Indian Election using Sentiment Analysis on Twitter (X) Data harnesses the power of sentiment analysis and Twitter data to forecast the outcome of the Indian election. By leveraging advanced natural language processing techniques, we analyze tweets related to Indian politics to discern public sentiment towards different political parties and candidates. Through this analysis, we aim to provide valuable insights into the prevailing mood of the electorate and predict the potential outcomes of the election. This project not only offers a novel approach to gauging public opinion but also presents an innovative method for political forecasting in the digital age. By combining the vast trove of Twitter data with sophisticated sentiment analysis algorithms, we strive to enhance the accuracy and timeliness of election predictions, thereby empowering stakeholders with valuable insights for strategic decision-making."""
        text2 = """In this project, we employ a comprehensive toolkit to analyze and interpret Twitter data for predicting the Indian election outcome. Leveraging Textblob for sentiment detection, we accurately classify tweets into positive, negative, or neutral sentiments, providing a nuanced understanding of public opinion dynamics. Additionally, we utilize Streamlit to develop an intuitive user interface, enabling stakeholders to interactively explore sentiment trends and prediction insights. To enhance the quality of our data analysis, we implement various preprocessing techniques such as tokenization, stop word removal, lemmatization, and stemming, ensuring the cleanliness and standardization of the tweet data. Finally, we showcase our findings and visualizations using a combination of Matplotlib and Plotly, offering both static and interactive graphical representations for easy interpretation and dissemination of results. Through this integrated approach, we aim to provide a robust and user-friendly platform for predicting the Indian election outcome based on sentiment analysis of Twitter data."""
        st.markdown(f"<div style='text-align: justify'>{text}</div>", unsafe_allow_html=True)
        st.image("sentiment.png", width=700)
        st.markdown(f"<div style='text-align: justify'>{text2}</div>", unsafe_allow_html=True)
        
    elif choice == "Text Analysis":
        st.subheader("Text Analysis")
        data = str()
        data= str(st.text_input("Enter the Sentence to analyze sentiment (Press Enter once done)"))
        if len(data) > 0 :
            data = analyze_sentiment(data)
            print(data)
            st.subheader("The result for Entered Data : " + data)
            
    elif choice == "Analysis File":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
            df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
            st.write("Total Positive Reviews are : {} | Total Negative Reviews are : {} | Total Neutral Reviews are : #{}".format(len(df[df["Sentiment"]=="Positive"]),len(df[df["Sentiment"]=="Negative"]), len(df[df["Sentiment"]=="Neutral"])))
            st.write(df)
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(df["Sentiment"]))
            st.pyplot(plt.gcf())
        

    elif choice == "Login":

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login/Logout"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                st.write("Press a 'Fetch and Predict' Button to get the sentiment analysis on :")
                
                
                
                st.sidebar.success("login Success.")
                if st.button("Fetch and Predict"):
                    analysis("Indian Election")
                    
            else:
                st.warning("Incorrect Username/Password")




    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()