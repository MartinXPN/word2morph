# WordSegmentation

### Usage - train a model
```commandline
python src/train.py
        fix_random_seed 0
        init_data --train_path datasets/rus.train --valid_path datasets/rus.dev
        construct_model --model_type CNN --embeddings_size 8 --kernel_sizes (5,5,5) --nb_filters (192,192,192) --recurrent_units (64,128,256) --dense_output_units 64 --dropout 0.2
        train --batch_size 32 --epochs 50 --patience 10 --log_dir logs --models_dir checkpoints
```

### Hyperparameter search (Bayesian tuning and bandits)
```commandline
python src/hyperparametersearch.py
        fix_random_seed 0
        init_data --train_path datasets/rus.train --valid_path datasets/rus.dev
        search_hyperparameters --nb_trials 50 --epochs 50 --patience 10 --log_dir logs --models_dir checkpoints
```

### Predict on test data
```commandline
python src/predict.py
        --model_path logs/<timestamp>/checkpoints/<modelname.hdf5> --processor_path logs/<timestamp>/processor.pkl
        --batch_size 80 --input_path path_to_input.txt --output_path path_to_output.txt
```
