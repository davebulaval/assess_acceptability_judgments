import os
import zipfile
from pathlib import Path
from typing import List, Optional, Union
from urllib.request import urlretrieve

import conllu
import nltk
import supar
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from supar import Parser

from .ressources import CACHE_PATH
from .util import DownloadProgressBar

CORENLP_URL = "http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip"


class DependencyParserCoreNLP:
    # Path to the corenlp JAR models to use for parsing and create Tree
    # As of july 2023, Stanza does not return a Tree by a dictionary. Thus, we use NLTK API
    # that parse and return a dependency parse tree.
    CORENLP_DIRECTORY = "stanford-corenlp-full-2018-02-27"
    JAR_FILE_NAME = os.path.join(CORENLP_DIRECTORY, "stanford-corenlp-3.9.1.jar")
    JAR_MODEL_FILE_NAME = os.path.join(CORENLP_DIRECTORY, "stanford-corenlp-3.9.1-models.jar")

    def __init__(self, verbose: bool = True, cache_path: Optional[str] = None) -> None:
        """
        Create a dependency parsing model that use CoreNLP dependency parser. To do so, we download the latest
        model from CoreNLP (i.e. 2018) as suggest by this Wiki
        https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK.

        :param verbose: (bool) Either or not to be verbose during the download of CoreNLP model.
        :param cache_path: (Optional[str]) Optional parameter to set a cache path to download the CoreNP model to.
            If the cache_path is not set, the model are downloaded in the default cache path i.e. `'.cache/aaj'`.
        """

        if cache_path is None:
            cache_path = CACHE_PATH

        self.jar_file_name = os.path.join(cache_path, self.JAR_FILE_NAME)
        self.jar_model_file_name = os.path.join(cache_path, self.JAR_MODEL_FILE_NAME)

        if not os.path.exists(self.jar_file_name) and not os.path.exists(self.jar_model_file_name):
            if verbose:
                reporthook = DownloadProgressBar()
            else:
                reporthook = None

            # Download zipped file with verbose report
            local_filename, _ = urlretrieve(CORENLP_URL, reporthook=reporthook)

            # Create .cache directory if it does not exist
            Path(cache_path).mkdir(parents=True, exist_ok=True)

            # Unzip the file into the cache directory
            with zipfile.ZipFile(local_filename, "r") as f:
                f.extractall(cache_path)

    def tree_parser_sentences(self, sentences: List[str]) -> List[List[Union[str, nltk.tree.tree.Tree]]]:
        """
        Method to parse sentences into dependency tree.

        :param sentences: (list) A list of sentence to parse into trees.
        :return: A list of NLTK dependency parse tree.
        """
        with CoreNLPServer(path_to_jar=self.jar_file_name, path_to_models_jar=self.jar_model_file_name) as server:
            parser = CoreNLPParser(url=server.url)
            parsed_trees = []
            for sentence in sentences:
                if len(sentence) > 0:
                    parsed_trees.append(list(parser.raw_parse(sentence)))
                else:
                    parsed_trees.append([""])
            return parsed_trees


class DependencyParserSuPar:
    def __init__(self, model: str):
        """
        Create a dependency parsing model that use SuPar constituency parser.

        Base on the SuPar documentation https://github.com/yzhangcs/parser#usage.

        :param model: (str) The parsing model to use. Choices are
            - `'biaffine'` (https://openreview.net/forum?id=Hk95PK9le),
            - `'crf'` (https://aclanthology.org/2020.acl-main.302/),
            - `'crf2o'` (https://aclanthology.org/2020.acl-main.302/), and
            - `'vi'` (https://aclanthology.org/2020.aacl-main.12).
        """

        self.process_pipeline = Parser.load(f'{model}-dep-en')

    def get_tree(self, sentence: supar.utils.Dataset) -> List[conllu.models.TokenTree]:
        """
        Interface method to get the tree depending on the sentence object.

        :param sentence: A SuPar Dataset.
        :return: Return a list of Tree SuPar Sentence.
        """
        return conllu.parse_tree(str(sentence.sentences[0]))

    def process_sentences(self, sentence: str) -> supar.utils.Dataset:
        """
        Interface method to process sentences.

        :param sentence: A sentence.
        :return: Return a SuPar dataset.
        """
        return self.process_pipeline.predict(sentence, lang="en", prob="False", verbose="False")

    def tree_parser_sentences(self, sentences: List[str]) -> List[List[Union[str, conllu.models.TokenTree]]]:
        """
        Method to parse sentences into constituency tree.

        :param sentences: (list) A list of sentence to parse into trees.
        :return: A list of SuPar parse tree.
        """
        parsed_trees = []

        for sentence in sentences:
            if len(sentence) > 0:
                process_documents = self.process_sentences(sentence)
                parsed_trees.append(self.get_tree(process_documents))
            else:
                parsed_trees.append([""])
        return parsed_trees
