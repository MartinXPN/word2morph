# WordSegmentation

### Usage
```commandline
python src/train.py
        fix_random_seed 0
        init_data --train_path datasets/rus.train --valid_path datasets/rus.dev
        construct_model --embeddings_size 8 --kernel_sizes (5,5,5) --nb_filters (192,192,192) --dense_output_units 64 --dropout 0.2
        train --batch_size 32 --epochs 50 --patience 10 --log_dir checkpoints --models_dir checkpoints
```
