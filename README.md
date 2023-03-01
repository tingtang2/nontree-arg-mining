# nontree-arg-mining
Implementation of the argument mining system from [Towards Better Non-Tree Argument Mining: Proposition-Level Biaffine Parsing with Task-Specific Parameterization](https://aclanthology.org/2020.acl-main.298/). 

Code heavy inspired by the [SuPar repository](https://github.com/yzhangcs/parser).

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```
Use of virtual environments is optional but recommended. For GPU support refer to the comments in the `requirements.txt` file.

After installing the libraries from the `requirements.txt` file also run the following command
to install the correct `spacy` pipeline:

```bash
python3 -m spacy download en_core_web_sm
```


Then change the `PATH_TO_DATASET` variable in the `run.py` file to the directory containing the CDCP dataset with transitive closure performed. This dataset is also available [here](https://tingtang2.github.io/files/CDCP_data.tar.gz). 
## Running the model

Please run the following command for information on the command line arguments:

```
python3 run.py -h
```

An example command for a training run using the validation set:
```
python3 run.py --epochs 50 --lr 12e-4 --elmo_embedding --glove_embedding --device 0 --save_dir ./morio-model-runs/ -train
```


An example command for evaluating on the test set: 
```
python3 run.py --cpu --checkpoint_path ./morio-model-runs/example.pt -test
```

## Repo structure

The `model.py` file contains all the PyTorch modules needed to construct the model while the `run.py` file consists of methods for loading, training, and evaluating, all orchestrated in `main()`.

## Issues

For any issues or comments please feel free to contact [Ting Chen](mailto:ting.chen@richmond.edu)

### Notes
- If you run into a 500 error from huggingface just wait a couple minutes and rerun it