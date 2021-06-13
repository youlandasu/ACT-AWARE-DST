## The Act-Aware Dialogue State Tracking Models

This is the PyTorch implementation of the following paper:
[Ruolin Su](https://github.com/youlandasu), Ting-Wei Wu, Biing-Hwang Juang. **Act-Aware Slot-Value Predicting in Multi-Domain Dialogue State Tracking.** *INTERSPEECH 2021.*

## How to Use
### Install Dependency
```
pip install -r requirements.txt
```

### Download and Create the MultiWOZ2.1 Dataset
```
wget https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/utils/mapping.pair
python create_data.py 
```

### Formatilize Dataset
```
python multiwoz_format.py all ./data ./data
```
### Elmo Embeddings
In environment `allennlp>=1.0.0` to calculate ELMO embeddings. See *requirements.txt* for details.
```
mkdir ./data/elmo_embeddings
./calc_elmo.sh ./data ./data/elmo_embeddings
```

### Train
1. Train our hybrid model (time- and num-related slots as non-categorical)
```
./train_sp_act.sh
```
2. Train our categorical-only model
```
./train_nosp_act.sh
```

### Evaluation
```
./predict.sh ./data/prediction_act.json
```
To evaluate on the dev set, change the target file to *./data/preddev_act.json*.


