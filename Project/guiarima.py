import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.express as px
from PIL import Image



st.set_page_config(page_title='Sales Forecasting')
st.header('Sales Forecasting')

st.subheader("Upload Data File")
data_file=st.file_uploader("Upload Sales File",type=["csv"])

if data_file is not None:
    forecasted_file = st.file_uploader("Upload Forecast File ", type=["csv"])
    df = pd.read_csv(data_file)
    if forecasted_file is not None:
        df2 = pd.read_csv(forecasted_file)
        df.dropna()
        st.dataframe(df, width=1000)
        df['Month'] = pd.to_datetime(df['Month'])
        df2['Month'] = pd.to_datetime(df2['Month'])

        df.set_index('Month', inplace=True)

        df2.set_index('Month', inplace=True)

        st.subheader('Graphical Representaion of Data ')
        st.line_chart(df, y='Monthly_Sales')


        def nonstationary():
            df['Seasonal First Difference'] = df['Monthly_Sales'] - df['Monthly_Sales'].shift(12)
            diffrence = 1
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(df['Monthly_Sales'], order=(1, diffrence, 1))
            model_fit = model.fit()
            end = len(df)
            df['Forecast'] = model_fit.predict(start=2, end=end, dynamic=True)
            import statsmodels.api as sm
            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, diffrence, 1),
                                              seasonal_order=(1, diffrence, 1, 12))
            results = model.fit()
            st.subheader("Accuracy of Model")
            df['Forecast'] = results.predict(start=end - 16, end=end, dynamic=True)
            st.line_chart(df, y=['Forecast', 'Monthly_Sales'])



        def stationary():
            diffrence = 0
            st.write("Non seasonal")
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(df['Monthly_Sales'], order=(1, diffrence, 1))
            model_fit = model.fit()
            end = len(df)
            df['Forecast'] = model_fit.predict(start=2, end=end, dynamic=True)
            import statsmodels.api as sm
            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, diffrence, 1),
                                              seasonal_order=(1, diffrence, 1, 12))
            results = model.fit()
            st.subheader("Accuracy of Model")
            df['Forecast'] = results.predict(start=end - 16, end=end, dynamic=True)
            st.line_chart(df, y=['Forecast', 'Monthly_Sales'])

            st.write("Mean Square error")


        def adfuller_test(sales):
            result = adfuller(sales)
            return result[1]


        z = adfuller_test(df['Monthly_Sales'])
        if (z <= 0.05):
            stationary()
            difference = 0
        else:
            nonstationary()
            difference = 1
        end = len(df) - 1

        st.subheader("Forecasted Data")
        slider = st.slider("Year", min_value=1, max_value=10, step=1)

        if (slider == 1):
            st.write('Forecasting for 1 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 12, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2,width=1000)

        if (slider == 2):
            st.write('Forecasting for 2 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 24, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 3):
            st.write('Forecasting for 3 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 36, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 4):
            st.write('Forecasting for 4 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 48, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 5):
            st.write('Forecasting for 5 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 60, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 6):
            st.write('Forecasting for 6 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 72, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 7):
            st.write('Forecasting for 7 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 84, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 8):
            st.write('Forecasting for 8 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 96, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 9):
            st.write('Forecasting for 9 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 120, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        if (slider == 10):
            st.write('Forecasting for 10 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=end + 132, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])
            st.dataframe(df2, width=1000)

        button1 = st.button("Forecasted Sale For Next 1 Year")
        button2 = st.button("Forecasted Sale For Next 2 Year")
        button3 = st.button("Forecasted Sale For Next 5 Year")
        button4 = st.button("Forecasted Sale For Next 10 Year")
        if button1:
            st.write('Forecasting for 1 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=121, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])

        if button2:
            st.write('Forecasting for 2 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=133, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])

        if button3:
            st.write('Forecasting for 5 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=169, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])

        if button4:
            st.write('Forecasting for 10 Year')
            import statsmodels.api as sm

            model = sm.tsa.statespace.SARIMAX(df['Monthly_Sales'], order=(1, difference, 1),
                                              seasonal_order=(1, difference, 1, 12))
            results = model.fit()
            df2['forecast'] = results.predict(start=end, end=229, dynamic=True)
            st.line_chart(df2, y=['forecast', 'Monthly_Sales'])


