import urllib.request
from pathlib import Path

import git
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, save_path: str, exists_ok: bool = True, verbose: int = 2):
    if url is None or save_path is None:
        raise ValueError('Both `url` and `save_path` need to be provided')

    if Path(save_path).exists():
        if not exists_ok:
            raise ValueError(f'File with the path `{save_path}` already exists')
        if verbose >= 1:
            print(f'File with the path `{save_path}` already exists')
        return

    description = url.split('/')[-1]
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description, disable=verbose < 2) as t:
        urllib.request.urlretrieve(url=url, filename=save_path, reporthook=t.update_to)


def get_current_commit() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
