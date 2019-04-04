# WordSegmentation

[![Build Status](https://travis-ci.com/MartinXPN/word2morph.svg?branch=master)](https://travis-ci.com/MartinXPN/word2morph)

### Usage - train a model
```commandline
PYTHONHASHSEED=0 python -m word2morph.train
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        construct_model --model_type CNN --embeddings_size 8 --kernel_sizes (5,5,5) --nb_filters (192,192,192) --recurrent_units (64,128,256) --dense_output_units 64 --dropout 0.2
        train --batch_size 32 --epochs 75 --patience 10 --log_dir logs
```

### Hyperparameter search (Bayesian tuning and bandits)
```commandline
PYTHONHASHSEED=0 python -m word2morph.hyperparametersearch
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        search_hyperparameters --nb_trials 50 --epochs 75 --patience 10 --log_dir logs
```

### Predict on test data
```commandline
PYTHONHASHSEED=0 python -m word2morph.predict
        --model_path logs/<timestamp>/checkpoints/<modelname.joblib>
        --batch_size 1 --input_path path_to_input.txt --output_path path_to_output.txt
```


## Pypi package
```commandline
python setup.py upload
```


#### TODO
* Evaluate results after post-processing
* Improve post-processing step