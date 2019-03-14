import pickle
from typing import Iterable, List, Tuple, Union

import fire
import numpy as np
from keras import Model
from keras.engine.saving import load_model
from tqdm import tqdm

from word2morph.data.generators import DataGenerator
from word2morph.data.loaders import DataLoader
from word2morph.data.processing import DataProcessor
from word2morph.entities.dataset import Dataset
from word2morph.entities.sample import Sample
from word2morph.models.cnn import CNNModel
from word2morph.models.rnn import RNNModel
from word2morph.util.metrics import Evaluate


class Word2Morph(object):
    def __init__(self, model_path: str, processor_path: str):
        self.model: Model = load_model(filepath=model_path, custom_objects={'CNNModel': CNNModel, 'RNNModel': RNNModel})
        with open(processor_path, 'rb') as f:
            self.processor: DataProcessor = pickle.load(file=f)

    def predict(self, inputs: Union[str, Iterable[Sample]], batch_size: int) -> List[Sample]:
        """
        :param inputs: either a string to a file or List of Sample-s
        :param batch_size: batch size in which to process the data
        :return: Predicted samples in the order they were given as an input
        """
        if type(inputs) == str:
            inputs = DataLoader(file_path=inputs).load()

        dataset: Dataset = Dataset(samples=inputs)
        predicted_samples: List[Sample] = []
        for batch_start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[batch_start: batch_start + batch_size]
            inputs, _ = self.processor.parse(batch, convert_one_hot=False)
            res: np.ndarray = self.model.predict(x=inputs)

            predicted_samples += [self.processor.to_sample(word=sample.word, prediction=prediction)
                                  for sample, prediction in zip(batch, res)]
        return predicted_samples

    def evaluate(self, inputs: Union[str, Iterable[Sample]], batch_size: int) -> Tuple[List[Tuple[Sample, Sample]],
                                                                                       List[Tuple[Sample, Sample]],
                                                                                       List[Sample]]:
        """
        :param inputs: either a string to a file or List of Sample-s
        :param batch_size: batch size in which to process the data
        :return: (list of correct predictions, list of wrong predictions, list of all predictions in the input order)
                    each list item is (predicted_sample, correct_sample)
        """
        if type(inputs) == str:
            inputs = DataLoader(file_path=inputs).load()

        ''' Show Evaluation metrics '''
        data_generator: DataGenerator = DataGenerator(dataset=Dataset(samples=inputs),
                                                      processor=self.processor,
                                                      batch_size=batch_size)
        evaluate = Evaluate(data_generator=iter(data_generator), nb_steps=len(data_generator), prepend_str='test_')
        evaluate.model = self.model
        evaluate.on_epoch_end(epoch=0)

        ''' Predict the result and print '''
        predicted_samples = self.predict(inputs=inputs, batch_size=batch_size)
        correct, wrong = [], []
        for correct_sample, predicted_sample in zip(inputs, predicted_samples):
            if predicted_sample == correct_sample:
                correct.append((predicted_sample, correct_sample))
            else:
                wrong.append((predicted_sample, correct_sample))
        print('Word accuracy after filtering only valid combinations:', len(correct) / len(inputs), flush=True)
        return correct, wrong, predicted_samples


def predict(model_path: str, processor_path: str, batch_size: int = 1,
            input_path='datasets/rus.test', output_path='logs/rus.predictions'):
    word2morph = Word2Morph(model_path=model_path, processor_path=processor_path)
    correct, wrong, predicted_samples = word2morph.evaluate(inputs=input_path, batch_size=batch_size)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(sample) for sample in predicted_samples]))


if __name__ == '__main__':
    fire.Fire(predict)
