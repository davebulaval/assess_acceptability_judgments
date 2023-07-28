from typing import List, Union, Generator

import benepar
import spacy
import stanza
import supar
from stanza.models.constituency.tree_reader import read_trees
from supar import Parser


# Todo: add encoding of the parse tree using Tree-LSTM: https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/README.md
# similar to https://www.hindawi.com/journals/cin/2022/4096383/


class ConstituencyParserCoreNLP:
    def __init__(self) -> None:
        """
         Create a dependency parsing model that use Stanza constituency parser.

        Base on the Stanza documentation https://stanfordnlp.github.io/stanza/constituency.html#simple-code-example.
        """

        self.process_pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    def get_tree(self, sentence: stanza.models.common.doc.Sentence) -> stanza.models.constituency.parse_tree.Tree:
        """
        Interface method to get the tree depending on the sentence object.

        :param sentence: A Stanza Sentence.
        :return: Return a Stanza Tree.
        """
        return sentence.constituency

    def process_sentences(self, sentences: List[str]) -> List[stanza.Document]:
        """
        Interface method to process sentences.

        :param sentences: A list of sentences.
        :return: Return a list of Stanza document.
        """
        return self.process_pipeline.bulk_process(sentences)

    def tree_parser_sentences(
        self, sentences: List[str]
    ) -> List[List[Union[str, stanza.models.constituency.parse_tree.Tree]]]:
        """
        Method to parse sentences into constituency tree.

        :param sentences: (list) A list of sentence to parse into trees.
        :return: A list of Stanza parse tree.
        """
        process_documents = self.process_sentences(sentences)

        parsed_trees = []
        for doc in process_documents:
            if len(doc.sentences) > 0:
                doc_parsed_trees = []
                for sentence in doc.sentences:
                    parsed_tree = self.get_tree(sentence)
                    doc_parsed_trees.append(parsed_tree)
                parsed_trees.append(doc_parsed_trees)
            else:
                parsed_trees.append([""])

        return parsed_trees


class ConstituencyParsingSuPar:
    def __init__(self, model: str) -> None:
        """
        Create a dependency parsing model that use SuPar constituency parser.

        Base on the SuPar documentation https://github.com/yzhangcs/parser#usage.

        :param model: (str) The parsing model to use. Choices are
            # - `'aj'` (https://papers.nips.cc/paper/2020/hash/f7177163c833dff4b38fc8d2872f1ec6-Abstract.html),
            - `'crf'` (https://www.ijcai.org/Proceedings/2020/560/),
            - `'tt'` (https://aclanthology.org/2020.acl-main.557), and
            - `'vi'` (https://aclanthology.org/2020.aacl-main.12).
        """

        self.process_pipeline = Parser.load(f'{model}-con-en')

    def get_tree(self, sentence: supar.utils.Dataset) -> List[supar.utils.transform.TreeSentence]:
        """
        Interface method to get the tree depending on the sentence object.

        :param sentence: A SuPar Dataset.
        :return: Return a list of Tree SuPar Sentence.
        """
        return sentence.sentences

    def process_sentences(self, sentence: str) -> supar.utils.Dataset:
        """
        Interface method to process sentences.

        :param sentence: A sentence.
        :return: Return a SuPar dataset.
        """
        return self.process_pipeline.predict(sentence, lang="en", prob="False", verbose="False")

    def tree_parser_sentences(self, sentences: List[str]) -> List[List[Union[str, supar.utils.transform.TreeSentence]]]:
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


class ConstituencyParsingBeNePar:
    def __init__(self, use_larger_model: bool = False) -> None:
        """
        Create a dependency parsing model that use BeNePar constituency parser.

        Base on the BeNePar documentation
        https://github.com/nikitakit/self-attentive-parser#usage-with-spacy-recommended.

        :param use_larger_model: (bool) either or not to use the larger model version. Larger model tak
            more RAM/GPU RAM than smaller one. See SpaCy and BeNePar documentation for details.
        """

        if use_larger_model:
            spacy_model = "en_core_web_trf"
            benepar_model = "benepar_en3_large"
        else:
            spacy_model = "en_core_web_md"
            benepar_model = "benepar_en3"

        spacy.cli.download(spacy_model)
        benepar.download(benepar_model)
        self.process_pipeline = spacy.load(spacy_model)
        self.process_pipeline.add_pipe("benepar", config={"model": benepar_model})

    def get_tree(self, sentence: spacy.tokens.Span) -> stanza.models.constituency.parse_tree.Tree:
        """
        Interface method to get the tree depending on the sentence object.

        :param sentence: A SpaCy Span.
        :return: Return a Stanza Tree.
        """

        return read_trees(sentence._.parse_string)

    def process_sentences(self, sentences: List[str]) -> spacy.Language.pipe:
        """
        Interface method to process sentences.

        :param sentences: A list of sentences.
        :return: Return a generator.
        """
        return self.process_pipeline.pipe(sentences)

    def tree_parser_sentences(
        self, sentences: List[str]
    ) -> List[List[Union[str, stanza.models.constituency.parse_tree.Tree]]]:
        """
        Method to parse sentences into constituency tree.

        :param sentences: (list) A list of sentence to parse into trees.
        :return: A list of Stanza parse tree.
        """

        process_documents = self.process_sentences(sentences)

        parsed_trees = []
        for process_document in process_documents:
            if len(process_document.text) > 0:
                doc_parsed_trees = []
                for sent in process_document.sents:
                    parsed_tree = self.get_tree(sent)
                    doc_parsed_trees.append(parsed_tree)
                parsed_trees.append(doc_parsed_trees)
            else:
                parsed_trees.append([""])
        return parsed_trees
