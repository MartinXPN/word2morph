from unittest import TestCase

from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping


class TestCharMappings(TestCase):
    def test_text_char_mapping(self):
        mapping = CharToIdMapping(text='zyxabc123', include_unknown=False)
        self.assertIn('z', mapping.key_to_id)
        self.assertNotIn(mapping.UNK, mapping.key_to_id)
        self.assertEqual(mapping['z'], 0)
        self.assertEqual(mapping['y'], 1)
        self.assertEqual(mapping['3'], 8)
        with self.assertRaises(KeyError):
            _ = mapping['k']
        with self.assertRaises(KeyError):
            _ = mapping[mapping.UNK]

    def test_char_mapping(self):
        mapping = CharToIdMapping(chars=list('abcd'), include_unknown=True)
        self.assertIn('a', mapping.key_to_id)
        self.assertIn(mapping.UNK, mapping.key_to_id)
        self.assertIn('c', mapping.key_to_id)
        self.assertEqual(mapping[mapping.UNK], 0)
        self.assertEqual(mapping['a'], 1)
        self.assertEqual(mapping['e'], mapping[mapping.UNK])
        self.assertEqual(mapping['f'], mapping[mapping.UNK])

    def test_word_char_mapping(self):
        mapping = CharToIdMapping(words=['hello', 'world'], chars=list('yo'), include_unknown=True)
        self.assertIn('h', mapping.key_to_id)
        self.assertIn(mapping.UNK, mapping.key_to_id)
        self.assertIn('y', mapping.key_to_id)
        self.assertIn('o', mapping.key_to_id)
        self.assertEqual(mapping['k'], mapping[mapping.UNK])


class TestWordMappings(TestCase):
    # упасти	у:PREF/пас:ROOT/ти:SUFF
    def test_word_segment_mapping(self):
        mapping = WordSegmentTypeToIdMapping(['PREF', 'ROOT', 'SUFF'], include_unknown=True)
        self.assertEqual(mapping['ROOT'], 2)
        self.assertEqual(mapping[mapping.UNK], 0)
        self.assertEqual(mapping['UNKNOWN'], mapping[mapping.UNK])
