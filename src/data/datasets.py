from typing import List


class Dataset(object):
    def __init__(self, file_path: str) -> None:
        self.data = self.load_data(file_path)

    @staticmethod
    def load_data(file_path: str) -> List[str]:
        with open(file_path) as f:
            return f.readlines()

    def __len__(self) -> int:
        return len(self.data)
