# WordSegmentation

### Usage
```commandline
python src/train.py \
        fix_random_seed 0 \
        init_data --train_path datasets/rus.train --valid_path datasets/rus.dev \
        construct_model \
        train --batch_size 16 --patience 10
```