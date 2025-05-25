import pandas as pd
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader

import yfinance as yf

# note regex import
import re

# Part Get Yahoo Finance Data
# todo dat에 따라서 end 컬럼의 시간이 바뀌어야 함.
StockData = yf.download('^NDX', start='2000-01-01', end='2025-05-25')

# print(StockData)

# Column : index, date , price, close, high, low, open, volume
StockData.to_csv('StockData_Analysis.csv')

# print(StockData)

# Get Data Collection for news title
# loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
# title_data = loader.load()

loader = WebBaseLoader('https://finviz.com/news.ashx')
title_data = loader.load()

df = title_data[0]

# list - 시간 \n 기사 제목
# print(title_data2)

# Part Regular expression page_content
test = re.findall(r'((?:\b(?:0?\d|1[0-2]):[0-5]\d[AP]M)|(?:\b[A-Za-z]{3}-\d{1,2}))[\n ]+([^\n]+)', title_data[0].page_content)

print(test)

exit()
# Part Load Data CSV file
testfile = pd.read_csv('UpStock-Analysis.csv')

testfile['date'] = test[0]

# note Regular expression DType : List[Tuple[str, str]]
print(test[0])

# todo 이제 csv 파일에 삽입해야 하는데 어떤식으로 저장하는게 더 좋을까? < 날짜 Column과 제목 Column을 추가하도록하자. 우선 test


exit()