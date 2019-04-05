# Digital-Signal-Processing
This repo include HMM hard coding realization, and typing disambiguity.

## Runing
HMM for example

#### Install
```bash
git clone
cd Digital-Signal-Processing/HMM
make
```

#### Train
./train [train iteration] [initial model path] [training data] [output model path]
```bash
./train 5 data/model_init.txt data/seq_model_01.txt data/model_01.txt
```

#### Evaluation
./test [model list path] [testing data] [output]
```bash
./test data/modellist.txt data/testing_data1.txt result1.txt
```


