import io
import numpy as np
import pickle

import fire
from keras import Model
from keras.engine.saving import load_model
from tqdm import tqdm

from src.data.loaders import DataLoader
from src.data.processing import DataProcessor
from src.entities.dataset import BucketDataset, Dataset
from src.models.cnn import CNNModel
from src.models.rnn import RNNModel


def predict(model_path: str, processor_path: str, batch_size: int = 80,
            input_path='datasets/rus.test', output_path='logs/rus.predictions'):
    model: Model = load_model(filepath=model_path, custom_objects={'CNNModel': CNNModel, 'RNNModel': RNNModel})
    dataset: Dataset = BucketDataset(samples=DataLoader(file_path=input_path).load())
    with open(processor_path, 'rb') as f:
        processor: DataProcessor = pickle.load(file=f)

    predicted_samples = []
    for batch_start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.data[batch_start: batch_start + batch_size]
        inputs, _ = processor.parse(batch, convert_one_hot=False)
        res: np.ndarray = model.predict(x=inputs)
        for sample, prediction in zip(batch, res):
            sample = processor.to_sample(word=sample.word, prediction=prediction)
            predicted_samples.append(sample)

    with io.open(output_path, 'w', encoding='utf-8') as f:
        for sample in predicted_samples:
            f.write(sample.word + '\t' + '/'.join([s.segment + ':' + s.type for s in sample.segments]) + '\n')


if __name__ == '__main__':
    fire.Fire(predict)
