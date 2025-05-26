import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
import yfinance as yf
import numpy as np

# note regex import
import re


# Get Yahoo Finance Data
StockData = yf.download('^NDX', start='2009-02-14', end='2025-05-25')

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

# open 가격 때보다 close 가격이 높으면 1 else 0
StockData['label'] = np.where( StockData['Open'] < StockData['Close'], 1, 0)

StockData.to_csv('DataSets/stock_price_data.csv')