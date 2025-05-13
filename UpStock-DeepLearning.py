import pandas as pd
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader

import yfinance as yf

# note regex import
import re

# Part Get Yahoo Finance Data
StockData = yf.download('^NDX', start='2000-01-01', end='2025-05-13')

# print(StockData)


# Part Save CSV file

# info Column : index, date , price, close, high, low, open, volume
StockData.to_csv('StockData_Analysis.csv')

# Part Get Data Collection for news title
# loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
# title_data = loader.load()

loader2 = WebBaseLoader('https://finviz.com/news.ashx')
title_data2 = loader2.load()


df = title_data2[0]

# list - 시간 \n 기사 제목
print(title_data2)

# Part 정규식 page_content 뒤에 나오는 내용, 


exit()

# Part Insert Data CSV (Supabase)

# Part Load Data CSV file

# Part 