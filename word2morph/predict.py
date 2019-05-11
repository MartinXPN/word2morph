import fire

from word2morph import Word2Morph
from word2morph.data.beamsearch import PostProcessBeamSearch
from word2morph.data.loaders import DataLoader


def predict(model_path: str, batch_size: int = 1,
            input_path='datasets/rus.test', output_path='logs/rus.predictions'):
    word2morph = Word2Morph.load_model(path=model_path)
    word2morph.processor = PostProcessBeamSearch(char_mapping=word2morph.processor.char_mapping,
                                                 word_segment_mapping=word2morph.processor.word_segment_mapping,
                                                 bmes_mapping=word2morph.processor.bmes_mapping,
                                                 label_mapping=word2morph.processor.label_mapping)
    inputs = DataLoader(file_path=input_path).load()
    correct, wrong, predicted_samples = word2morph.evaluate(inputs, batch_size=batch_size)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(sample) for sample in predicted_samples]))


if __name__ == '__main__':
    fire.Fire(predict)
