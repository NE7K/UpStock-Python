import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
import yfinance as yf
import numpy as np

# note regex import
import re

# Get Yahoo Finance Data
StockData = yf.download('^NDX', start='2025-09-05', end='2025-09-09')

print(StockData)
