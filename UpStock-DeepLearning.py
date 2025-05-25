import pandas as pd
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader

import yfinance as yf

# note regex import
import re

# Get Yahoo Finance Data
StockData = yf.download('^NDX', start='1971-02-05', end='2025-05-25')
StockData.to_csv('StockData_Analysis.csv')

# Get CNBC news
# loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
# title_data = loader.load()

# csv file load
# Stock_PriceData = pd.read_csv('StockData_Analysis.csv')

loader = WebBaseLoader('https://finviz.com/news.ashx')
title_data = loader.load()

df = title_data[0].page_content

# 날짜와 내용 분리 정규식
pattern = r"(May-\d{2}|Apr-\d{2}|Jun-\d{2})\s+([^\n]+)"
regex_title = re.findall(pattern, df)

df = pd.DataFrame(regex_title, columns=['date', 'news_context'])    
df.to_csv('UpStock-NewsData.csv', index=False)

# df = pd.read_csv('test.csv')
print(df)
