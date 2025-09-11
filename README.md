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
## 📁 Data Set 1 : Dataset used in the sentiment model

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

## 📁 Data Set 2

```
```

## Sentiment Model Training result and layers status

```
Epoch 1/20
loss: 0.6352 - accuracy: 0.6516 - val_loss: 0.5148 - val_accuracy: 0.7498

Epoch 2/20
loss: 0.3841 - accuracy: 0.8314 - val_loss: 0.4343 - val_accuracy: 0.8007

Epoch 3/20
loss: 0.1724 - accuracy: 0.9359 - val_loss: 0.5599 - val_accuracy: 0.7869

Epoch 4/20
loss: 0.0806 - accuracy: 0.9743 - val_loss: 0.7665 - val_accuracy: 0.7826

Epoch 5/20
loss: 0.0433 - accuracy: 0.9853 - val_loss: 0.9666 - val_accuracy: 0.7627

loss: 0.0433 - accuracy: 0.9852
Restoring model weights from the end of the best epoch: 2.
Epoch 5: early stopping
```

## Sentiment Model summary()

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 model_input (InputLayer)    [(None, 141)]             0         
                                                                 
 embedding (Embedding)       (None, 141, 128)          1304192   
                                                                 
 bidirectional (Bidirection  (None, 141, 128)          98816     
 al)                                                             
                                                                 
 global_max_pooling1d (Glob  (None, 128)               0         
 alMaxPooling1D)                                                 
                                                                 
 dense (Dense)               (None, 64)                8256      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 1413377 (5.39 MB)
Trainable params: 1413377 (5.39 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

## Sentiment Model Predict
```
1/1 [==============================] - 0s 281ms/step
[negative] EM portfolios funnel near $45 billion in August but cracks are showing, IIF says
 : 0.52

[positive] Stocks' Bull Market Nears 3-Year Anniversary. It Likely Has More Room to Run.
 : 0.87

[negative] Stock Market Today: Dow Slides As Oracle Soars; Medicare News Hits Health Leader
 : 0.19

[negative] Stock Market Today: Dow and Nasdaq fall, S&P 500 loses momentum ahead of August consumer-price index on Thursday; Oracle share surge highlights technology spending
 : 0.67

[positive] Oracle stock booms 35%, on pace for best day since 1992
 : 0.86

```


## System configuration diagram

<img width="100%" alt="system diagram" src="https://github.com/user-attachments/assets/1fe24cfe-0da4-4d8e-bc22-7ada93908529" />

## System Detail Diagram

<img width="100%" alt="UserFlow" src="https://github.com/user-attachments/assets/ed7e5638-796c-4820-8bde-9148ea4b2da3" />

## 🔍 WBS

-


## 📧 Contact

For questions or feedback, please contact [NE7K](mailto:sjjang16@naver.com) or [NeighborSoft](mailto:neighborsoft@gmail.com).
