<img width="100%" src="https://github.com/user-attachments/assets/ca28df82-b7fe-467c-a86e-113321e0b5a9" width="100%" height="100%"> </img>

## 📖 Overview

미국 증시는 소셜 미디어의 발달로 인해 정보 확산 속도가 극도로 빨라졌으며, 단일 뉴스 기사나 트윗이 단기적인 변동성을 유발하는 사례가 빈번하게 발생하고 있습니다. 이에 본 프로젝트는 뉴스, 블로그, 댓글 등과 같은 비정형 데이터를 자연어 처리 기반 인공지능 모델을 활용해 분석함으로써, 시장 참여자들의 심리적 반응을 정량적으로 평가합니다.

더 나아가, 단순한 감성 분석을 넘어 시장 고평가/저평가 여부를 반영하는 VIX 변동성 지표, 과매수·과매도 판단을 위한 RSI 보조지표, 이동평균선 기반 추세 지표 등을 통합적으로 고려합니다. 이를 통해 시장 심리를 0에서 100 사이의 값으로 환산한 Market Sentiment Index를 산출합니다.

본 시스템은 전문 투자 지식이 부족한 개인 투자자뿐 아니라, 시간 제약으로 인해 시장 모니터링이 어려운 투자자들에게도 실질적인 의사결정 보조 도구로 기능할 수 있습니다.

The U.S. stock market has become highly sensitive to the rapid spread of information driven by the rise of social media, where a single news article or tweet can frequently trigger short-term volatility. To address this, the project leverages a natural language processing (NLP)–based artificial intelligence model to analyze unstructured data such as news articles, blogs, and comments, thereby quantifying market participants’ psychological responses.

Beyond sentiment analysis alone, the system also incorporates key market indicators, including the VIX volatility index to capture overvaluation and undervaluation, the RSI oscillator to identify overbought and oversold conditions, and moving averages to assess price trends. These elements are integrated to calculate a Market Sentiment Index, expressed as a numerical value ranging from 0 to 100.

This system is designed not only for individual investors with limited expertise in financial indicators but also for those constrained by time, providing a practical decision-support tool to better understand the prevailing psychological state of the U.S. stock market.

## Project file structure

```
upstock-python/
│── upstock/
│   │
│   ├── builders/
│   │   ├── pipeline.py            # pipline builder
│   │
│   ├── indicators/
│   │   ├── core.py                # calculation logic
│   │   ├── indexer.py             # Indicator indexing/management
│   │
│   ├── models/
│   │   ├── artifacts.py           # Model artifact management
│   │
│   ├── nodes/
│   │   ├── predict.py             # Predict node -> News predict part, market predict part
│   │   ├── train.py               # Train node
│   │
│   ├── storage/
│   │   ├── downloader.py          # Data download management
│   │   ├── market_data.py         # Stock Data Processing
│   │
│   ├── config.py                  # Environment
│
│── main.py                        # Main entry point
```

## 📁 Data Set : Dataset used in the sentiment model

| Text                                                                                                                     | Label |
|--------------------------------------------------------------------------------------------------------------------------|-------|
| ANA with the conference tomorrow, I guess we will hit 10.05 after all...                                                 | 1     |
| NFX - hearing PJCO out cautious following comScore data: Following Weak December, Overall 4Q Traffic Again Turns Negative user | 0     |
| se pullback 2 initiate quick trades, not invest, long. Keep stops tight & sell up. AAP EGN OCN AMH DDD NSM               | 1     |
| Piper Jaffray making negative comments on NFX and CST ed Box : comScore Q4 data weak for edbox, Netflix                  | 0     |
| Maybe a good time to buy TD the 44 level appears to be holding as a key support level                                    | 1     |
| CSN option trader buys 1,500 of the Jan 11-16 call spread against low OI indicating entering a position for .50. Bet on data bef. Jan 19 | 1     |
| AA there goes the kids college fund!                                                                                     | 1     |

- Author(s). *Dataset Title*. Kaggle. Published 2025.  
  doi: [10.34740/kaggle/dsv/1217821](https://doi.org/10.34740/kaggle/dsv/1217821)  
  License: Data files © Original Authors

## Sentiment Model Training result and history

| Epoch | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 0.6352 | 0.6516   | 0.5148   | 0.7498       |
| 2     | 0.3841 | 0.8314   | 0.4343   | 0.8007       |
| 3     | 0.1724 | 0.9359   | 0.5599   | 0.7869       |
| 4     | 0.0806 | 0.9743   | 0.7665   | 0.7826       |
| 5     | 0.0433 | 0.9853   | 0.9666   | 0.7627       |

**Early Stopping:** Best epoch → **2**

## Sentiment Model summary()

| Layer (type)          | Output Shape   | Param # |
|------------------------|----------------|---------|
| InputLayer             | (None, 141)    | 0       |
| Embedding              | (None, 141,128)| 1,304,192 |
| Bidirectional(LSTM)    | (None, 141,128)| 98,816  |
| GlobalMaxPooling1D     | (None, 128)    | 0       |
| Dense                  | (None, 64)     | 8,256   |
| Dropout                | (None, 64)     | 0       |
| Dense                  | (None, 32)     | 2,080   |
| Dense                  | (None, 1)      | 33      |

**Total params:** 1,413,377 (5.39 MB)  
**Trainable params:** 1,413,377  
**Non-trainable params:** 0

## System configuration diagram

<img width="100%" alt="system diagram" src="https://github.com/user-attachments/assets/b186c72f-2df9-402c-87bd-fe1d1a4d419f" />

## Predict result saved in Supabase Table

예측 결과는 News Sentiment와 Market Sentiment Index 테이블로 나누어서 저장되고 보관되고 있습니다.

### News Sentiment Table

```SQL
SELECT text, percent, label FROM news_sentiment LIMIT 10

```
| text                                                                                                                            | percent   | label    |
| ------------------------------------------------------------------------------------------------------------------------------- | --------- | -------- |
| Dow jumps 400 points to record as August inflation increase likely won't derail Fed rate cut                                    | 0.0548475 | negative |
| Dow jumps and S&P touches all-time high while Treasury yields fall as Wall Street reacts to rising inflation and jobless claims | 0.25916   | negative |
| The Fed’s 2% inflation target might not be hit for years, says Janus portfolio manager — unless there’s a recession             | 0.295348  | negative |
| Dow, S&P 500 and Nasdaq push deeper into record territory                                                                       | 0.88824   | positive |
| Stocks, Bonds Rise as Data Seal September Fed Cut: Markets Wrap                                                                 | 0.813471  | positive |
| Mexico hikes China auto tariff, S. Korea warns on US investment                                                                 | 0.141901  | negative |
| Stocks, Gold Set New Records in Run-Up to Fed Meet: Markets Wrap                                                                | 0.821933  | positive |
| Vietnam Urges US to Rethink Seafood Ban as Trade Talks Grind On                                                                 | 0.212265  | negative |
| Stocks climb toward record closes                                                                                               | 0.813072  | positive |
| Shell LNG Plant Wins Place on Carney’s List of Favored Projects                                                                 | 0.924645  | positive |

### Market Sentiment Index Table

```SQL
SELECT date_utc, score, zone, rsi, vix, macd_val FROM market_sentiment_index LIMIT 10
```
| date_utc   | score | zone    | rsi             | vix              | macd_val         |
| ---------- | ----- | ------- | --------------- | ---------------- | ---------------- |
| 2025-09-18 | 58    | Neutral | 67.892115296438 | 15.5600004196167 | 6.48942949393609 |

## 🖥️ Training environment

```
Tensorflow 2.10

Cpu : AMD 5600
Ram : 32GB
Gpu : RTX 4060 8GB
```

## WBS

<img width="90%" alt="WBS" src="https://github.com/user-attachments/assets/62f8b65d-6d24-4020-90f1-7ebe8901075f"/>

## Flow

<img width="90%" alt="Flow" src="https://github.com/user-attachments/assets/5c04838c-12d5-4c34-b190-b9749a112080"/>


## 📧 Contact

For questions or feedback, please contact [NE7K](mailto:sjjang16@naver.com) or [NeighborSoft](mailto:neighborsoft@gmail.com).
