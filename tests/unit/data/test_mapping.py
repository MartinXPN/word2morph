from unittest import TestCase

from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping


class TestCharMappings(TestCase):
    def test_text_char_mapping(self):
        mapping = CharToIdMapping(text='zyxabc123', include_unknown=False)
        self.assertIn('z', mapping.key_to_id)
        self.assertNotIn(mapping.UNK, mapping.key_to_id)

    def test_char_mapping(self):
        mapping = CharToIdMapping(chars=set('abcd'), include_unknown=True)
        self.assertIn('a', mapping.key_to_id)
        self.assertIn(mapping.UNK, mapping.key_to_id)
        self.assertIn('c', mapping.key_to_id)

    def test_word_char_mapping(self):
        mapping = CharToIdMapping(words=['hello', 'world'], chars=set('yo'), include_unknown=True)
        self.assertIn('h', mapping.key_to_id)
        self.assertIn(mapping.UNK, mapping.key_to_id)
        self.assertIn('y', mapping.key_to_id)
        self.assertIn('o', mapping.key_to_id)


class TestWordMappings(TestCase):
    # упасти	у:PREF/пас:ROOT/ти:SUFF
    def test_word_segment_mapping(self):
        mapping = WordSegmentTypeToIdMapping(['PREF', 'ROOT', 'SUFF'], include_unknown=True)
        self.assertEqual(mapping['ROOT'], 2)
        self.assertEqual(mapping[mapping.UNK], 0)
