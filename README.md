# nontree-arg-mining
Implementation of [Morio et. al's](https://aclanthology.org/2020.acl-main.298/) argument mining system

Code heavy inspired by the [SuPar repository](https://github.com/yzhangcs/parser).

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```
pip install -r requirements.txt
```
Use of virtual environments is optional but recommended.


Then change the `PATH_TO_DATASET` variable in the `run.py` file to the directory containing the CDCP dataset with transitive closure performed. This dataset is also available upon request. 
## Running the model

Please run the following command for information on the command line arguments:

```
python3 run.py -h
```

An example command for a training run using 3 fold cross validation:
```
python3 --epochs 50 --lr 12e-4 --elmo_embedding --glove_embedding --device 0 --save_dir ./morio-model-runs/ -train

```


An example command for evaluating on the test set: 
```
python3 --device 0 --model_path ./morio-model-runs/example.pt -test

```

## Repo structure

The `model.py` file contains all the PyTorch modules needed to construct the model while the `run.py` file consists of methods for loading, training, and evaluating, all orchestrated in `main()`.

## Issues

For any issues or comments please feel free to contact [Ting Chen](mailto:ting.chen@richmond.edu)