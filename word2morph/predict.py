import io
import numpy as np
import pickle

import fire
from keras import Model
from keras.engine.saving import load_model
from tqdm import tqdm

from word2morph.data.generators import DataGenerator
from word2morph.data.loaders import DataLoader
from word2morph.data.processing import DataProcessor
from word2morph.entities.dataset import BucketDataset, Dataset
from word2morph.models.cnn import CNNModel
from word2morph.models.rnn import RNNModel
from word2morph.util.metrics import Evaluate


def predict(model_path: str, processor_path: str, batch_size: int = 1,
            input_path='datasets/rus.test', output_path='logs/rus.predictions'):
    model: Model = load_model(filepath=model_path, custom_objects={'CNNModel': CNNModel, 'RNNModel': RNNModel})
    dataset: Dataset = BucketDataset(samples=DataLoader(file_path=input_path).load())
    with open(processor_path, 'rb') as f:
        processor: DataProcessor = pickle.load(file=f)

    ''' Show Evaluation metrics '''
    data_generator: DataGenerator = DataGenerator(dataset=dataset,
                                                  processor=processor,
                                                  batch_size=batch_size)
    evaluate = Evaluate(data_generator=iter(data_generator), nb_steps=len(data_generator), prepend_str='test_')
    evaluate.model = model
    evaluate.on_epoch_end(epoch=0)

    ''' Predict the result and print '''
    predicted_samples = []
    correct_words, wrong_words = [], []
    for batch_start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.data[batch_start: batch_start + batch_size]
        inputs, _ = processor.parse(batch, convert_one_hot=False)
        res: np.ndarray = model.predict(x=inputs)
        for sample, prediction in zip(batch, res):
            predicted_sample = processor.to_sample(word=sample.word, prediction=prediction)
            predicted_samples.append(predicted_sample)

            # Keep track of right and wrong predictions
            if sample == predicted_sample:
                correct_words.append((sample, predicted_sample))
            else:
                wrong_words.append((sample, predicted_sample))

    print('Word accuracy after filtering only valid combinations:', len(correct_words) / len(dataset), flush=True)
    with io.open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(sample) for sample in predicted_samples]))


if __name__ == '__main__':
    fire.Fire(predict)
