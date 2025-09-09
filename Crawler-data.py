import pandas as pd
import yfinance as yf
import numpy as np

# langchain import
from langchain_community.document_loaders import WebBaseLoader

# Get Yahoo Finance Data
StockData = yf.download('^NDX', start='2009-02-14', end='2020-06-12')

# create independence dataset 
StockData = StockData.copy()
# 인덱스 초기화
StockData.reset_index(inplace=True)

# Date -> date
StockData.rename(columns={'Date' : 'date'}, inplace=True)

# 멀티 플렉스 > 단일, 다른 개별 주식 이용하면 확장필요
StockData.columns = StockData.columns.droplevel(1)

# open 가격 때보다 close 가격이 높으면 1 else 0
StockData['label'] = np.where( StockData['Open'] < StockData['Close'], 1, 0)

# PART 로그 수익률, 로그 비율, 로그 거래량 칼럼 추가
# prev close 전날 종가
StockData['prev_Close'] = StockData['Close'].shift(1)
# 전일 대비 증가 로그수익률
StockData['Close_logret'] = np.log(StockData['Close'] / StockData['prev_Close'])
# 저가/종가 로그비율
StockData['Low_logret'] = np.log(StockData['Low'] / StockData['Close'])
# 고가/종가 로그비율
StockData['High_logret'] = np.log(StockData['High'] / StockData['Close'])
# 시가/종가 로그비율
StockData['Open_logret'] = np.log(StockData['Open'] / StockData['Close'])
# 로그 1+거래량
StockData['Volume_log'] = np.log1p(StockData['Volume'])

# 첫 raw의 prev_Close 제거
StockData = StockData.dropna(subset=['prev_Close']).reset_index(drop=True)

# save dataset
StockData.to_csv('DataSets/stock_price_data_v2.csv', index=False)

# TODO 현재 진행하지 않는 부분
# RSI, MACD
# StockData['Rsi'] = 

# Get CNBC news
# loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
# title_data = loader.load()

# csv file load
# Stock_PriceData = pd.read_csv('StockData_Analysis.csv')

# Part Finviz Crawle
# loader = WebBaseLoader('https://finviz.com/news.ashx')
# title_data = loader.load()

# df = title_data[0].page_content

# # 날짜와 내용 분리 정규식
# pattern = r"(May-\d{2}|Apr-\d{2}|Jun-\d{2})\s+([^\n]+)"
# regex_title = re.findall(pattern, df)

# df = pd.DataFrame(regex_title, columns=['date', 'news_context'])    
# df.to_csv('UpStock-NewsData.csv', index=False)

# df = pd.read_csv('test.csv')