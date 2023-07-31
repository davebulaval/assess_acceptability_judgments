# from AESLF.aeslf.fluency_score_interface import LanguageModelFluencyScoreInterface
#
# # https://github.com/opencog/link-grammar/blob/master/bindings/python/linkgrammar.py
# # https://github.com/opencog/link-grammar/tree/master/docker/docker-python
# # Ou en service Java https://github.com/opencog/link-grammar/blob/master/docker/docker-parser/Dockerfile
#
#
from typing import Optional, List, Union

import nltk

try:
    import linkgrammar as lg
    from linkgrammar import Sentence, ParseOptions, Dictionary
except ImportError:
    lg = None


class LinkGrammar:
    # todo fix documentation href etc.
    def __init__(self, verbose: bool = True, cache_path: Optional[str] = None) -> None:
        """
        Create a Link Grammar parsing model that use the OpenCog version of the CMU Link Grammar natural language
        parser https://github.com/opencog/link-grammar.
        Since download URL is flaky, we use the HTTP one as suggested in this issue
        https://github.com/opencog/link-grammar/issues/1472.

        :param verbose: (bool) Either or not to be verbose during the download of CoreNLP model.
        :param cache_path: (Optional[str]) Optional parameter to set a cache path to download the Link Grammar model to.
            If the cache_path is not set, the model are downloaded in the default cache path i.e. `'.cache/aaj'`.

        ..note
            README INSTALL

            Link Grammar can be challenging to install. Here we propose a step-by-step guide to install it.

                1. Download the archive here
                'http://www.abisource.com/downloads/link-grammar/5.12.0/link-grammar-5.12.0.tar.gz'
                2. Unzip the archive.
                3. Install on your OS the package 'flex' and 'libpcre2-dev', these are **not** Python pacakge.
                4. In a command line interface (CLI), move to the repository 'link-grammar-5.12.0'.
                5. In the CLI, do the command './configure --enable-python-bindings'.
                6. Verify that the Python path is the one you need (see 'Python interfaces:' line).
                7. In the CLI, do the command 'make'.
                8. In the CLI, do the command 'sudo make install' (it need administrator/sudo permission).
                9. In the CLI, do the command 'sudo ldconfig' (it need administrator/sudo permission).
        """

        if lg is None:
            raise ImportError("")

        self.en_dictionary = Dictionary()
        self.parse_options = ParseOptions(min_null_count=0, max_null_count=999)

    def tree_parser_sentences(self, sentences: List[str]) -> List[List[Union[str, nltk.tree.tree.Tree]]]:
        """
        Method to parse sentences into dependency tree.

        :param sentences: (list) A list of sentence to parse into trees.
        :return: A list of NLTK dependency parse tree.
        """
        for sentence in sentences:
            sent = Sentence(sentence, self.en_dictionary, self.parse_options)
            linkages = sent.parse()
            # linkage_stat(sent, 'English', linkages, po)
            # for linkage in linkages:
            #     desc(linkage)
            return parsed_trees
