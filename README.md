# Digital-Signal-Processing
This repo includes hard coding realization about HMM and typing disambiguity.

## Runing
HMM for example

#### Install
```bash
git clone git@github.com:vachelch/Digital-Signal-Processing.git
cd Digital-Signal-Processing/HMM
make
```

#### Train

```bash
# ./train [train iteration] [initial model path] [training data] [output model path]
./train 5 data/model_init.txt data/seq_model_01.txt data/model_01.txt
```

#### Evaluation
```bash
# ./test [model list path] [testing data] [output]
./test data/modellist.txt data/testing_data1.txt result1.txt
```


