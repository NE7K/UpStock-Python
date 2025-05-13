import pandas as pd
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader

import yfinance as yf

# Part Get Yahoo Finance Data
StockData = yf.download('^NDX', start='2000-01-01', end='2025-05-13')

print(StockData)

exit()

# Part Save CSV file

# Part Get Data Collection for news title
loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
title_data = loader.load()

loader2 = WebBaseLoader('https://finviz.com/news.ashx')
title_data2 = loader2.load()


print(title_data)
print(title_data2)

# Part Insert Data CSV (Supabase)

# Part Load Data CSV file

# Part 