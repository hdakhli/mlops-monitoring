from unittest import TestCase

from train.train_model import ingest_data, preprocess_data


class Test(TestCase):
    def test_preprocess_data(self):
        # Given
        file_path = "train/data/titanic.xls"
        titanic = ingest_data(file_path)

        # When
        preprocessed = preprocess_data(titanic)

        # Then
        self.assertEqual("int64", str(preprocessed['sex'].dtype))
