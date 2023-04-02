# FATE: Few-shot Anomaly detection in TExt

## Setup guide:

First, install the requirements:
```
pip install -r requirements.txt
```

To generate AG News dataset:
```
cd datasets
python ag.py
bash generate_outliers_ag.sh
```

To generate 20Newgroups dataset:
```
cd datasets
python 20ng.py
bash generate_outliers_20ng.sh
```

To generate Reuters-21578 dataset:
```
cd datasets
python reuters.py
```

To generate contaminated datasets, run:
```
cd datasets
python contaminate_ds.py
```

## Running experiments:
To train and test the model:

```
cd src
python main.py --dataset ag --inlier_class 0
```

Choose the dataset from ["ag", "20ng", "reuters"].