import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie


st.set_page_config(page_title= "PRICE PREDICTION APP", page_icon=":chart_with_upwards_trend:", layout = "wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

# -- LOAD ANIMATION ASSETS --
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_l5qvxwtf.json")
lottie_email = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_vpzw63hs.json")
lottie_logo = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_qwATcU.json")


start = '2010-01-01'
end = '2019-12-31'


st.title('ONLINE STOCK MARKET PRICE PREDICTION SYSTEM')
st_lottie(lottie_logo, height = 300, key = "logo") #Animation
st.write("The goal of this website is to help investors :briefcase: make sound investment decisions :rocket: :full_moon_with_face: by studying the prediction chart :bar_chart: ")
# scraping data from yahoo finance

user_input = st.text_input('Enter stock ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Data from 2010 - 2019')
        st.write(df.describe())
    with right_column:
        st_lottie(lottie_coding, height = 300, key = "Analyze") #Animation
        


#visualizations

st.line_chart(df.Close)


st.subheader('Closing Price :closed_lock_with_key: vs Time chart :hourglass:')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close, label = 'Closing Price')
plt.legend()
st.pyplot(fig)



st.subheader('Closing Price :closed_lock_with_key: vs Time chart :hourglass: with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'r', label = '100 day MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price :closed_lock_with_key: vs Time chart :hourglass: with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'r', label = '100 day MA')
plt.plot(ma200, 'g', label = '200 day MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.legend()
st.pyplot(fig)


# Data is split into 70% training and 30% testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)


# load model
model = load_model('keras_model.h5')

#testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final graph
st.subheader('Prediction :satellite: vs Original price')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# -- Contact -- 
with st.container():
    st.write("---")
    st.header("Get in touch with Me! :smile: ")
    st.write("##")
    
    contact_form = """
    <form action="https://formsubmit.co/mohamedsalat62@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html = True)
    with right_column:
        st_lottie(lottie_email, height = 300, key = "email") #Animation