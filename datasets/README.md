# Datasets for word segmentation

The datasets include `train`, `valid` and `test` samples for several languages.

#### List of datasets with examples from the data
* English (`radioactivity`	`radio/act/iv/ity`)
* Finnish (`harrastuksessaan`	`harrast/ukse/ssa/an`)
* Turkish (`kullanImIndan`	`kul/lan/Im/I/ndan`)
* Russian (`гидротурбостроение`	`гидр:ROOT/о:LINK/турб:ROOT/о:LINK/стро:ROOT/ени:SUFF/е:END`)


#### Acknowledgement
The data is taken from the repository 
[NeuralMorphemeSegmentation](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation "Maintained by Alexey Sorokin").

The original train files were split into train and validation sets with 0.2 validation split
```python
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data, test_size=0.2)
```

And the `.dev` files were renamed to `.test` to be used for testing the models.