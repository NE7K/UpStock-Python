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

## System configuration diagram

<img width="100%" alt="system diagram" src="https://github.com/user-attachments/assets/1fe24cfe-0da4-4d8e-bc22-7ada93908529" />

## 🔍 WBS

-

## 📊 Target User Flow

The following diagram illustrates the user flow within the UpStock application :

<img width="100%" alt="UserFlow" src="https://github.com/user-attachments/assets/4379bea8-8f86-4f87-b6f6-7e2da1a93823" />
    
## 📝 Contribution Guide

1. Fork this repository.
2. Create a new branch. (`git checkout -b feature/AmazingFeature`)
3. Commit your changes. (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch. (`git push origin feature/AmazingFeature`)
5. Open a pull request.

## 🚀 Installation and Setup

```bash
# Clone the repository
git clone https://github.com/NE7K/upstock.git

# Move to the project directory
cd upstock

# Install dependencies
flutter pub get

# Run the application
flutter run
```

## 📧 Contact

For questions or feedback, please contact [NE7K](mailto:sjjang16@naver.com) or [NeighborSoft](mailto:neighborsoft@gmail.com).
