# WordSegmentation

### Usage - train a model
```commandline
python -m src.train
        fix_random_seed 0
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        construct_model --model_type CNN --embeddings_size 8 --kernel_sizes (5,5,5) --nb_filters (192,192,192) --recurrent_units (64,128,256) --dense_output_units 64 --dropout 0.2
        train --batch_size 32 --epochs 75 --patience 10 --log_dir logs
```

### Hyperparameter search (Bayesian tuning and bandits)
```commandline
python -m src.hyperparametersearch
        fix_random_seed 0
        init_data --train_path datasets/rus.train --valid_path datasets/rus.valid
        search_hyperparameters --nb_trials 50 --epochs 75 --patience 10 --log_dir logs
```

### Predict on test data
```commandline
python -m src.predict
        --model_path logs/<timestamp>/checkpoints/<modelname.hdf5> --processor_path logs/<timestamp>/processor.pkl
        --batch_size 80 --input_path path_to_input.txt --output_path path_to_output.txt
```


#### TODO
* Evaluate results after post-processing
* Improve post-processing step