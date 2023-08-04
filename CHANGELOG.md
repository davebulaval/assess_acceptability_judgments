## dev

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
