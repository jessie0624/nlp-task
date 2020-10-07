from os import unlink
from pathlib import Path
from typing import Dict, List, Tuple
from typing import Union

from gensim.corpora.dictionary import Dictionary
from pydantic import BaseModel

from nlp.data.const import Const 

class VocabConfig(BaseModel):
    names: List[str]

class Vocab:
    CONFIG_NAME = "config.json"

    def __init__(self, vocabs: Dict[str, Dictionary], pad_id=0, unk_id=1):
        self._vocabs = vocabs
        self._pad_id = pad_id
        self._unk_id = unk_id
    
    @classmethod
    def load_from_text(cls, directory: Path) -> "Vocab":
        config = VocabConfig.parse_file(directory / cls.CONFIG_NAME)
        vocabs = {}
        for name in config.names:
            path = str(directory / (name + '.txt'))
            vocabs[name] = Dictionary.load_from_text(path)
        return cls(vocabs)
    
    def from_documents(self, name: str, documents: List[List[str]]) -> "Vocab":
        if isinstance(documents[0], str):
            documents = [[doc] for doc in documents]
        vocab = Dictionary.from_documents(documents)
        self.vocabs[name] = vocab
        return self
    
    @property
    def vocabs(self):
        return self._vocabs
    
    @property
    def pad_id(self):
        return self._pad_id
    
    @property
    def unk_id(self):
        return self._unk_id
    
    @property
    def label_vocab(self):
        return self.get_vocab(Const.label)
    
    def get_vocab(self, name: str) -> Dictionary:
        if name in self.vocabs.keys():
            return self.vocabs[name]
        else:
            raise KeyError(f"can't find {name}, PS: {self.vocabs.keys()}")

    def save_as_text(self, directory: Path) -> None:
        config = VocabConfig(names=list(self.vocabs.keys()))
        with open(str(directory / self.CONFIG_NAME), "W") as f:
            f.write(config.json())
        
        for name, vocab in self.vocabs.items():
            path = str(directory / (name + ".txt"))
            vocab.save_as_text(path, sort_by_word=False)
    
    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None) -> None:
        for vocab in self.vocabs.values():
            vocab.filter_extremes(no_below, no_above, keep_n, keep_tokens)

    def filter_n_most_frequent(self, remove_n: int) -> None:
        for vocab in self.vocabs.values():
            vocab.filter_n_most_frequent(remove_n)

    def filter_tokens(self, bad_ids=None, good_ids=None) -> None:
        for vocab in self.vocabs.values():
            vocab.filter_tokens(bad_ids, good_ids)

    def compactify(self) -> None:
        for vocab in self.vocabs.values():
            vocab.compactify()

    def add_documents(self, name: str, documents: List[List[str]]) -> None:
        vocab = self.get_vocab(name)
        vocab.add_documents(documents)

    def doc2bow(
        self, name: str, document: List[str], allow_update=False, return_missing=False
    ) -> List[Tuple[int, int]]:
        vocab = self.get_vocab(name)
        return vocab.doc2bow(document, allow_update, return_missing)

    def doc2idx(self, name: str, document: Union[str, List[str]], unknown_word_index=None) -> List[int]:
        unk_id = unknown_word_index if unknown_word_index is not None else self.unk_id
        vocab = self.get_vocab(name)
        if isinstance(document, str):
            return vocab.token2id.get(document, unk_id)
        else:
            return vocab.doc2idx(document, unk_id)

    def idx2doc(self, name: str, ids: Union[int, List[int]]) -> List[str]:
        vocab: Dictionary = self.get_vocab(name)
        if isinstance(ids, int):
            return [vocab[ids]]
        else:
            tokens = []
            for id_ in ids:
                tokens.append(vocab[id_])
            return tokens    