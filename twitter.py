import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import mplfinance as fplt
import matplotlib as mpl
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import plotly.graph_objects as go
import plotly.express as px

#------------------------------load model-----------------------------------#
with(open("save_models/scale_pipe_3_features.pkl", "rb")) as file:
    preprocess = pickle.load(file)

with(open("save_models/var_scale.pkl", "rb")) as file:
    var_scale= pickle.load(file)

model = tf.keras.models.load_model('save_models/lstm_5_3_features.h5')

#------------------------------twitter preprocessing--------------------------#
def demoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'',text)

def cleaning(text):
    text = demoji(text)
    text = re.sub(r'http\S+',' ', text)  #remove urls
    text = re.sub(r'@\w+',' ', text) #remove mentions
    text = re.sub(r'#\w+', ' ', text) #remove hashtag
    text = re.sub('[^A-Za-z\']+', ' ', text) #remove characters that not use in the english alphabets
    text = text.lower() #lower caps
    text = re.sub('\w*\d\w*','', text) #remove digit

    return text

def generate_impact_score(compound_tweet) : 
    coef_verified = 2 if compound_tweet.user_verified else 1
    log_follower = math.log10((compound_tweet.user_followers)+1)
    impact_score = coef_verified * (log_follower+1)
    return impact_score

def generate_impact_vad(compound_tweet) : 
    coef_verified = 2 if compound_tweet.user_verified else 1
    log_follower = math.log10((compound_tweet.user_followers)+1)
    impact_score = (coef_verified * (log_follower+1) * compound_tweet.compound)+0.2
    return impact_score
#------------------------------load data-----------------------------------#
df = pd.read_csv('inference/BTCUSDT-1h-2022-02.csv', header=None)
df_sentiment = pd.read_csv('dataset/sampling_impact_score_all.csv', index_col=0)
df = df.rename(columns={0: 'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6: 'close_time', 7: 'quote_asset_volume', 8: 'number_of_trades', 9: 'taker_buy_asset_volume', 10: 'taker_buy_quote_asset_volume', 11: 'ignore'})


df['open_time_conv'] = pd.to_datetime(df['open_time'], unit='ms')

df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_sentiment['days'] = pd.to_datetime(df_sentiment['date']).dt.date
df_sentiment_avg = df_sentiment.drop_duplicates(subset='days', keep='first')
daterange = pd.date_range("2022-02-01", "2022-02-28")
for date in daterange:
    if date not in df_sentiment_avg['days'].values:
        df_sentiment_avg = df_sentiment_avg.append({'days': date, 'impact_score_var_avg': 0, 'impact_score_tb_avg': 0}, ignore_index=True)
df_sentiment_avg = df_sentiment_avg[['impact_score_var_avg', 'impact_score_tb_avg', 'days']]
df_sentiment_avg['days'] = pd.to_datetime(df_sentiment_avg['days']).dt.date
df_sentiment_avg.sort_values(by='days', inplace=True)
df_sentiment_avg.reset_index(drop=True, inplace=True)
df_sentiment_avg = df_sentiment_avg.sort_values(by='days')
df_sentiment_avg.reset_index(drop=True, inplace=True)
df_merge = df.copy()
df_merge['days'] = pd.to_datetime(df_merge['open_time_conv']).dt.date
df_merge = df_merge.merge(df_sentiment_avg, on='days', how='left')
df_merge.dropna(inplace=True)

#-----------twitter information----------#
st.title('Twitter Information')

col1, col2 = st.columns(2)

with col1:
    fol1= st.number_input('Insert a number of followers', 0, step=1, key="1")

with col2:
    ver1= st.selectbox('Is a Account is verified or not',('Yes','No'), key="2")

text1 = st.text_input('Input Tweets')

col3, col4 = st.columns(2)
with col3:
    fol2 = st.number_input('Insert a number of followers', 0, step=1, key="3")

with col4:
    ver2= st.selectbox('Is a Account is verified or not',('Yes','No'), key="4")

text2 = st.text_input('Input Tweets', key="5")

col5, col6 = st.columns(2)
with col5:
    fol3= st.number_input('Insert a number of followers', 0, step=1, key="6")

with col6:
    ver3=st.selectbox('Is a Account is verified or not',('Yes','No'), key="7")

text3 = st.text_input('Input Tweets', key="8")

#------------Inputasi Price-----------#
st.title('Price Information')

op, cp= st.columns(2)

with op:
    open_p = st.number_input('Open Price', 0.00, value=43500.00, step=0.01)
with cp:
    close_p = st.number_input('Close Price', 0.00, value=44000.00, step=0.01)


hp, lp, vl = st.columns(3)

with hp:
    high_p = st.number_input('High Price', 0.00, value=44500.00, step=0.01)
with lp:
    low_p = st.number_input('Low Price', 0.00, value=43000.00, step=0.01)
with vl:
    volume_p = st.number_input('volume', 0.00, value=50.00, step=0.01)

predict = st.button('Predict!')

#-----------price information------------#

# col1, col2 = st.columns(2)
# col1.metric('Open Price','1000','1200')
# col2.metric('Close Price','2000','-1000')

# col1, col2, col3 = st.columns(3)
# col1.metric('High',10) 
# col2.metric('Low',10) 
# col3.metric('Volume',10) 

#--------------------Prediction--------------------------------------------#
row_inf = 7 * 24
df_merge.loc[len(df_merge)] = close_p
df_merge['MA5'] = df_merge['close'].rolling(window=5).mean()
df_merge['impact_score_var_avg'].iloc[len(df_merge)-1] = 0
df_merge.dropna(inplace=True)
df_selected = df_merge[len(df_merge)-row_inf:].loc[:, ['close', 'MA5', 'impact_score_var_avg']]
df_scaled = preprocess.transform(df_selected.values[len(df_selected)-5:])
df_scaled_pred = df_scaled.reshape(-1, 5, 3)

# y_pred = model.predict(df_scaled)
# y_pred = preprocess.inverse_transform(y_pred.reshape(-1, 1))
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('Predicted Close Prices:')
if predict==True:
    y_pred = model.predict(df_scaled_pred)
    y_pred_real = np.concatenate([y_pred, y_pred, y_pred], axis=1)
    y_pred_real = preprocess.inverse_transform(y_pred_real)
    y_pred_real = y_pred_real[:, 0]

    last_close_price = df_merge['close'].iloc[len(df_merge)-1]
    close_predicted_price = y_pred_real[0].item()
    col1, col2 = st.columns(2)
    col1.metric('Close Price Prediction', '${0:,}'.format(np.round(close_predicted_price, 2)), f"{np.round(((close_predicted_price - last_close_price)*100/last_close_price), 2)}%")
    col2.metric('Last Close Price', '${0:,}'.format(last_close_price))
    # Visualization of the First Three Data
    df_real = preprocess.inverse_transform(df_scaled)
    df_real = df_real[:, 0]
    def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$price$"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(series, ".-", color="#E91D9E")
        if y is not None:
            plt.plot(row_inf+1, y, "bx", markersize=10, color="blue")
        if y_pred is not None:
            plt.plot(row_inf+1, y_pred, "ro")
        plt.grid(True)
        if x_label:
            plt.xlabel(x_label, fontsize=16,)
        if y_label:
            plt.ylabel(y_label, fontsize=16)
        st.pyplot(fig)
    show_7_days = df_merge['close'].iloc[len(df_merge)-row_inf:]
    show_7_days = show_7_days.reset_index(drop=True)
    plot_series(show_7_days, y_pred_real[0])
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.title('BTC/USDT Performance')
    last_24_price = df_merge['close'].iloc[len(df_merge)-24].item()
    all_24_price = df_merge['close'].iloc[len(df_merge)-24:]
    all_24 = df_merge.iloc[len(df_merge)-24:]
    all_720 = df_merge.iloc[:-1]
    st.subheader(f'Last 24h Price: ${last_24_price}')
    col1, col2, col3 = st.columns(3)
    col1.metric('24H Change(Ago)', '${0:,}'.format(np.round(last_close_price - last_24_price, 2)), f"{np.round(((last_close_price - last_24_price)*100/last_24_price), 2)}%")
    col2.metric('24H High', '${0:,}'.format(all_24_price.max()) )
    col3.metric('24H Low', '${0:,}'.format(all_24_price.min()))

    col1, col2 = st.columns(2)
    # col1.metric('24H Change(Avg)', '${0:,}'.format(np.round(last_close_price - all_24_price.mean(), 2)), f"{np.round(((all_24_price.mean() - last_close_price)*100/last_close_price), 2)}%")
    col1.metric('24H Volume(USDT)', '${0:,}'.format(np.round(all_24['quote_asset_volume'].sum(),1)))
    col2.metric('24H Volume(BTC)', '{0:,}'.format(np.round(all_24['volume'].sum())))   
    
    #Plot
    # Removing the localization in 'close_time'
    # df['close_time'] = df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_plot = df_merge.iloc[len(df_merge)-row_inf:-1]
    df_plot['open_time_conv'] = pd.to_datetime(df['open_time_conv'], errors='coerce')
    df_plot.index = pd.DatetimeIndex(df_plot['open_time_conv'])
    
    fig, ax = fplt.plot(
                df_plot,
                figsize=(20,5),
                type='candle',
                style='yahoo',
                title='BTC - 2017 ~ 2022',
                ylabel='Price ($)',
                volume=True,
                ylabel_lower='Shares\nTraded',
                show_nontrading=True,
                datetime_format='%Y-%m-%d',
                mav=(5,10),
                returnfig=True)
    
    st.pyplot(fig)
    st.info("Notes:\nBlue Lines: 5 MA, Orange Lines: 10 MA")

    #---------------------Twitter Preprocessing----------------------------------------#
    data = [text1, fol1, ver1]

    columns = ['text', 'user_followers', 'user_verified']
    new_data = pd.DataFrame([data], columns=columns)
    new_data.loc[len(new_data)] = ([text2, fol2, ver2])
    new_data.loc[len(new_data)] = ([text3, fol3, ver3])

    sid = SentimentIntensityAnalyzer()
    new_data['cleaned_text'] = new_data['text'].apply(cleaning)
    new_data['vader'] = new_data['cleaned_text'].apply(lambda desc: sid.polarity_scores(desc))
    new_data['compound'] = new_data['vader'].apply(lambda score_dict: score_dict['compound'])

    new_data['impact_score_only'] = new_data.apply(generate_impact_score,axis=1)

    new_data['impact_score_var'] = new_data.apply(generate_impact_vad,axis=1)
    new_data['impact_score_var_avg'] = new_data['impact_score_var'].mean()
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.title('Sentiment Evaluation')
    sentimen_result = new_data['impact_score_var_avg'].iloc[0]
    sentimen_eval = np.where(sentimen_result > 0.66, 'Positive 😎', np.where(sentimen_result < -0.66, 'Negative 🤒', 'Neutral/Unclear 🙂')).item()
    col1, col2 = st.columns(2)
    neutral_exp = "Seems like there's no impactful tweets \nthat might bring effects on the market."
    negative_exp = "This might bring negative effects on the market \nespecially if this happening for several days. \nBe cautious with your trade!" 
    positive_exp = "This might bring positive effects on the market \n, but always keeping an eye and aware with real market conditions."
    sentimen_exp = np.where(sentimen_result > 0.66, positive_exp, np.where(sentimen_result < -0.66, negative_exp, neutral_exp)).item()
    with col1:
        col1.metric('Sentiments For Today', sentimen_eval)
        col1.write(sentimen_exp)
    
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader('By Volume:')
    col1, col2 = st.columns(2)
    with col1:
        d30_avg = all_720.groupby('days').sum()['quote_asset_volume'].mean()
        col1.metric('24H Volume(USDT)', '${0:,}'.format(np.round(all_24['quote_asset_volume'].sum(),1)),  f"{np.round(((all_24['quote_asset_volume'].sum() - d30_avg)*100/d30_avg), 2)}%")
    with col2:
        
        col2.metric('30D AVG Volume(USDT)', '${0:,}'.format(np.round(d30_avg,1)))
        # col2.metric('24H Volume(BTC)', '{0:,}'.format(np.round(all_24['volume'].sum())))
    
    col1, col2 = st.columns(2)
    with col1:
        d30_avg = all_720.groupby('days').sum()['quote_asset_volume'].mean()
        last_2days = all_720.groupby('days').sum()['quote_asset_volume'].iloc[-2]
        col1.metric('Today with Last Day Differences', '${0:,}'.format(np.round(all_24['quote_asset_volume'].sum() - last_2days,1)),  f"{np.round(((all_24['quote_asset_volume'].sum() - last_2days)*100/last_2days), 2)}%")
    with col2:
        col2.metric('30D Standard Deviation', '${0:,}'.format(np.round(all_720.groupby('days').sum()['quote_asset_volume'].std(),1)))
    
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader('By Volatility:')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        d30_avg = all_720.groupby('days').mean()['close'].mean()
        col1.metric('24H Change(Ago)', '${0:,}'.format(np.round(last_close_price - last_24_price, 2)), f"{np.round(((last_close_price - last_24_price)*100/last_24_price), 2)}%")
    with col2:
        col2.metric('30D AVG Change', '${0:,}'.format(np.round(last_close_price - d30_avg, 2)), f"{np.round(((last_close_price - d30_avg)*100/d30_avg), 2)}%")
    with col3:
        col3.metric('30D Average Price', '${0:,}'.format(np.round(d30_avg, 2)))

    condition1 = var_scale.transform(np.array(new_data['impact_score_var']).reshape(-1,1))
    minmax_vol = MinMaxScaler()
    minmax_vol.fit(np.array(all_720.groupby('days').sum()['quote_asset_volume']).reshape(-1,1))
    condition2 = minmax_vol.transform(all_24['quote_asset_volume'].sum().reshape(-1,1))
    if sentimen_eval == 'Negative 🤒':
        condition2 = (1 - condition2)
    minmax_price = MinMaxScaler()
    condition3 = minmax_price.fit(np.array(all_720.groupby('days')['close'].mean()).reshape(-1,1))
    condition3 = minmax_price.transform(last_close_price.reshape(-1,1))
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    market_result = (condition2[len(condition2)-1] + condition3[len(condition3)-1])/2
    overall_result = (condition1.mean() + market_result)/2
    st.subheader('Current Market Condition:')
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = np.round(overall_result[0], 2)*100,
        gauge = {'axis': {'range': [0, 100]},
                'steps' : [
                    {'range': [0, 33.3], 'color': "#ff7003"},
                    {'range': [33.3, 66.67], 'color': "#ffe603"},]},
        domain = {'x': [0, 1], 'y': [0, 1]},),
        )

    st.plotly_chart(fig)
    # st.write(all_720.groupby('days').mean()['close'])
else:
    st.subheader("--")