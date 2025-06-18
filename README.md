<img src="https://github.com/user-attachments/assets/ca28df82-b7fe-467c-a86e-113321e0b5a9" width="100%" height="100%"> </img>

## 📖 Project Overview

Analysis and Summary of US Stock Market Sentiment Indicators and Market Data

## 🖥️ Training environment

```
Tensorflow 2.10

Cpu : AMD 5600
Ram : 32GB
Gpu : RTX 4060 8GB
```
## 📁 Data Set 1 : Dataset used in the initial model

Column : label, low, high, close, open, volume, title

| Index   | Title                                                 | Date       | Stock | Low     | Open    | Volume     | Label |
| ------- | ----------------------------------------------------- | ---------- | ----- | ------- | ------- | ---------- | ----- |
| 0       | Stocks That Hit 52Week Highs On Friday                | 2020-06-05 | A     | 9659.45 | 9673.09 | 6607730000 | 1     |
| 1       | Stocks That Hit 52Week Highs On Wednesday             | 2020-06-03 | A     | 9648.93 | 9689.72 | 4679030000 | 1     |
| 2       | 71 Biggest Movers From Friday                         | 2020-05-26 | A     | 9379.90 | 9570.53 | 4448950000 | 0     |
| 3       | 46 Stocks Moving In Fridays MidDay Session            | 2020-05-22 | A     | 9325.07 | 9363.67 | 3668070000 | 1     |
| 4       | B of A Securities Maintains Neutral on Agilent...     | 2020-05-22 | A     | 9325.07 | 9363.67 | 3668070000 | 1     |
| 1373579 | Top Narrow Based Indexes For August 29                | 2011-08-29 | ZX    | 2187.72 | 2188.67 | 1615510000 | 1     |
| 1373580 | Recap Wednesdays Top Percentage Gainers and Losers    | 2011-06-22 | ZX    | 2234.83 | 2243.21 | 1617370000 | 0     |
| 1373581 | UPDATE Oppenheimer Color on China Zenix Auto Industry | 2011-06-21 | ZX    | 2203.71 | 2210.97 | 1882490000 | 1     |
| 1373582 | Oppenheimer Initiates China Zenix At Outperform       | 2011-06-21 | ZX    | 2203.71 | 2210.97 | 1882490000 | 1     |
| 1373583 | China Zenix Auto International Opens For Trading      | 2011-05-12 | ZX    | 2372.19 | 2384.14 | 2209650000 | 1     |



## 📁 Data Set 2 : Dataset used in later models

ex)
```json
{"Unnamed: 0": 1413841, "headline": "China Zenix Announces Open Market Stock Purchases of 15K Shares by Management", "url": "https://www.benzinga.com/news/11/12/2233145/china-zenix-announces-open-market-stock-purchases-of-15k-shares-by-management", "publisher": "Eddie Staley", "date": "2011-12-30 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413842, "headline": "China Zenix Auto International Awarded Wheel Supply Contract From Sany Heavy Industry; Terms Not Disclosed", "url": "https://www.benzinga.com/news/11/11/2153607/china-zenix-auto-international-awarded-wheel-supply-contract-from-sany-heavy-indu", "publisher": "Allie Wickman", "date": "2011-11-28 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413843, "headline": "China Zenix Auto International Selected as Exclusive Wheel Producer to Chery-CIMC Truck JV", "url": "https://www.benzinga.com/news/11/11/2142369/china-zenix-auto-international-selected-as-exclusive-wheel-producer-to-chery-cimc", "publisher": "Eddie Staley", "date": "2011-11-21 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413844, "headline": "Top Narrow Based Indexes For August 29", "url": "https://www.benzinga.com/news/11/08/1888782/top-narrow-based-indexes-for-august-29", "publisher": "Monica Gerson", "date": "2011-08-29 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413845, "headline": "Recap: Wednesday's Top Percentage Gainers and Losers", "url": "https://www.benzinga.com/news/earnings/11/06/1193660/recap-wednesdays-top-percentage-gainers-and-losers", "publisher": "Benjamin Lee", "date": "2011-06-22 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413846, "headline": "UPDATE: Oppenheimer Color on China Zenix Auto Initiation", "url": "https://www.benzinga.com/analyst-ratings/analyst-color/11/06/1186890/update-oppenheimer-color-on-china-zenix-auto-initiation", "publisher": "BenzingaStaffL", "date": "2011-06-21 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413847, "headline": "Oppenheimer Initiates China Zenix At Outperform, $8 PT", "url": "https://www.benzinga.com/analyst-ratings/price-target/11/06/1186025/oppenheimer-initiates-china-zenix-at-outperform-8-pt", "publisher": "Joe Young", "date": "2011-06-21 00:00:00", "stock": "ZX"}
{"Unnamed: 0": 1413848, "headline": "China Zenix Auto International Opens For Trading at $6.00; IPO Price Set at $6.00", "url": "https://www.benzinga.com/news/ipos/11/05/1078911/china-zenix-auto-international-opens-for-trading-at-6-00-ipo-price-set-at-6-", "publisher": "Allie Wickman", "date": "2011-05-12 00:00:00", "stock": "ZX"}
```

## Training result and layers status

```
34340/34340 [==============================] - 1073s 31ms/step - loss: 0.3354 - accuracy: 0.8275 - val_loss: 0.1118 - val_accuracy: 0.9781
Epoch 2/50
34340/34340 [==============================] - 1103s 32ms/step - loss: 0.1983 - accuracy: 0.9155 - val_loss: 0.1211 - val_accuracy: 0.9481
Epoch 3/50
34340/34340 [==============================] - 1114s 32ms/step - loss: 0.1747 - accuracy: 0.9270 - val_loss: 0.1039 - val_accuracy: 0.9582
Epoch 4/50
34340/34340 [==============================] - 1097s 32ms/step - loss: 0.1548 - accuracy: 0.9372 - val_loss: 0.1306 - val_accuracy: 0.9385
Epoch 5/50
34340/34340 [==============================] - 1104s 32ms/step - loss: 0.1356 - accuracy: 0.9455 - val_loss: 0.1020 - val_accuracy: 0.9627
Epoch 6/50
34340/34340 [==============================] - 1103s 32ms/step - loss: 0.1250 - accuracy: 0.9504 - val_loss: 0.1365 - val_accuracy: 0.9373
Epoch 7/50
34340/34340 [==============================] - 1105s 32ms/step - loss: 0.1153 - accuracy: 0.9547 - val_loss: 0.0913 - val_accuracy: 0.9686
Epoch 8/50
34340/34340 [==============================] - 1114s 32ms/step - loss: 0.1079 - accuracy: 0.9577 - val_loss: 0.1228 - val_accuracy: 0.9530
Epoch 9/50
34340/34340 [==============================] - 1111s 32ms/step - loss: 0.1017 - accuracy: 0.9605 - val_loss: 0.1415 - val_accuracy: 0.9391
Epoch 10/50
34340/34340 [==============================] - 1098s 32ms/step - loss: 0.0930 - accuracy: 0.9640 - val_loss: 0.0887 - val_accuracy: 0.9686
Epoch 11/50
34340/34340 [==============================] - 1095s 32ms/step - loss: 0.0895 - accuracy: 0.9657 - val_loss: 0.1168 - val_accuracy: 0.9578
Epoch 12/50
34340/34340 [==============================] - 1087s 32ms/step - loss: 0.0856 - accuracy: 0.9670 - val_loss: 0.1324 - val_accuracy: 0.9569
Epoch 13/50
34339/34340 [============================>.] - ETA: 0s - loss: 0.0814 - accuracy: 0.9689Restoring model weights from the end of the best epoch: 10.
34340/34340 [==============================] - 1096s 32ms/step - loss: 0.0814 - accuracy: 0.9689 - val_loss: 0.1181 - val_accuracy: 0.9597
Epoch 13: early stopping

```

```
Total params: 27,699,344
Trainable params: 27,699,329
Non-trainable params: 15
```

## System configuration diagram

<img width="100%" alt="system diagram" src="https://github.com/user-attachments/assets/1fe24cfe-0da4-4d8e-bc22-7ada93908529" />

## System Detail Diagram

<img width="100%" alt="UserFlow" src="https://github.com/user-attachments/assets/ed7e5638-796c-4820-8bde-9148ea4b2da3" />

## 🔍 WBS

-


## 📧 Contact

For questions or feedback, please contact [NE7K](mailto:sjjang16@naver.com) or [NeighborSoft](mailto:neighborsoft@gmail.com).
