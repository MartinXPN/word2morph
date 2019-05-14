# WordSegmentation

[![Build Status](https://travis-ci.com/MartinXPN/word2morph.svg?branch=master)](https://travis-ci.com/MartinXPN/word2morph)


### Prerequisites
* Python 3.6
* Clone the repository and install the dependencies
```bash
git clone https://github.com/MartinXPN/word2morph.git
cd word2morph
pip install .
```

### Train a model
```bash
PYTHONHASHSEED=0 python -m word2morph.train
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        construct_model --model_type CNN --embeddings_size 8 --kernel_sizes '(5,5,5)' --nb_filters '(192,192,192)' --dilations '(1,1,1)' --recurrent_units '(64,128,256)' --use_crf=True --dense_output_units 64 --dropout 0.2
        train --batch_size 32 --epochs 75 --patience 10 --log_dir logs
```

### Hyperparameter search (Bayesian tuning and bandits)
```bash
PYTHONHASHSEED=0 python -m word2morph.hyperparametersearch
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        search_hyperparameters --nb_trials 50 --epochs 100 --patience 10 --log_dir logs
```

### Predict on test data
```bash
PYTHONHASHSEED=0 python -m word2morph.predict
        --model_path logs/<timestamp>/checkpoints/best-model.joblib
        --batch_size 1 --input_path path_to_input.txt --output_path path_to_output.txt
```
