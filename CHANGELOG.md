## Beta (0.0.27)

- Add missing tag in tagset of Conll-U.


## Beta (0.0.26)

- Fix CoreNLPClient call in `ConstituencyParserCoreNLP`. It seems like the context manager does not work will with
  multiple initialization and call. Now instead, we instantiate and start the client in `__init__` and use it in the for
  loop and close it on `__dell__`.

## Beta (0.0.25)

- Fix CoreNLPClient call in `ConstituencyParserCoreNLP`. We were creating the server in a for loop which is faster that
  the Client init thus it was creating multiple client instead of one.

## Beta (0.0.24)

- Change the approach to pass endpoint and start_server as keyword arguments in the parse method
  for `ConstituencyParserCoreNLP`.

## Beta (0.0.23)

- `ConstituencyParserCoreNLP` has now the start_server option to start if not already started. Based
  on [this](https://github.com/stanfordnlp/stanza/issues/810).

## Beta (0.0.22)

- `ConstituencyParserCoreNLP` can now be set on a different endpoint.

## Beta (0.0.21)

- `ConstituencyParserCoreNLP` is REALLY quiet now.

## Beta (0.0.20)

- `ConstituencyParserCoreNLP` is always quiet now.

## Beta (0.0.19)

- Fix negated verbose in `ConstituencyParserCoreNLP`.
- Add missing tags in `ConstituencyParserCoreNLP`.

## Beta (0.0.18)

- Now use Stanza CoreNLP interface to allow binary tree parsing in `ConstituencyParserCoreNLP`.

## Beta (0.0.17)

- Add missing tags in `ConstituencyParserCoreNLP`.

## Beta (0.0.16)

- Add missing tags in `ConstituencyParserCoreNLP`.

## Beta (0.0.15)

- Add missing tags in `ConstituencyParserCoreNLP`.

## Beta (0.0.14)

- BugFix interface naming (`CoreNLPParser`) that was colliding with NLTK same class name.

## Beta (0.0.13)

- Fix `DependencyParserCoreNLP` `tree_parser_sentences` method that was not returning proper graph.
- Add parsed tree tags to `DependencyParserCoreNLP` and `ConstituencyParserCoreNLP`.

## Beta (0.0.12)

- Add TQDM into `tree_parser_sentences` methods.

## Beta (0.0.11)

- Fix the error in `DependencyParserCoreNLP` that returns a constituency tree rather than a dependency tree.
- `ConstituencyParserCoreNLP` now uses CoreNLP like `DependencyParserCoreNLP` instead of Stanza.

## Beta (0.0.10)

- Fix import problem with CACHE.

## Beta (0.0.9)

- Refactor parser name that was using "ing" instead or "er".
- Fix import problem with parsers.

## Beta (0.0.8)

- Add dependency parser and constituency parser in alpha version. For now, they do not compute a score but only create
  the parse tree.

## Beta (0.0.7)

- Fix max sequence length bug for NLTK tokenizer and improve robustness of LLM tokenizer.

## Beta (0.0.6)

- Improve scores compute robustness for NaN value.

## Beta (0.0.5)

- Improves sentence len compute to be more robust when the batch comprises only empty strings.

## Beta (0.0.4)

- Fix resources path
- Fix error in `compute_lm` typing

## Beta (0.0.3)

- Add proper resources

## Beta (0.0.2)

- Fix PyLint project name

## Beta (0.0.1)

- Initial release of the library
- Implement SLOR, WPSLOR, NCE, PPL
