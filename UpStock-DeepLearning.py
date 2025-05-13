import pandas as pd
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader


# Part Get Data Collection for news title
loader = WebBaseLoader('https://www.cnbc.com/world/?region=world')
title_data = loader.load()

loader2 = WebBaseLoader('https://finviz.com/news.ashx')
title_data2 = loader2.load()


print(title_data)
print(title_data2)