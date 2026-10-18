# Corpus sources and preparation

Author: John Muir. Source pages checked on 2026-10-18 UTC.
The local corpus was prepared during the project's October 2026 experiments.

| Work | Project Gutenberg release date | Source |
| --- | --- | --- |
| My First Summer in the Sierra | 2010-05-26 | https://www.gutenberg.org/ebooks/32540 |
| Travels in Alaska | 2005-01-01 | https://www.gutenberg.org/ebooks/7345 |
| The Mountains of California | 2003-11-01 | https://www.gutenberg.org/ebooks/10012 |

These are Gutenberg release dates, not the books' original publication dates.
Each source page lists the work as public domain in the USA. Source digitization
credits are provided on those pages. Project Gutenberg's license is available
at https://www.gutenberg.org/policy/license.html . The project's MIT license
covers project code and documentation, not ownership of the source books or
Project Gutenberg branding. Check applicable terms when redistributing source
editions, especially outside the USA.

## What was removed

The first book was trimmed to Chapter I through the end of the narrative,
excluding Gutenberg headers/footers, the index, and bracketed illustrations.
The two added books were trimmed from their first narrative paragraphs to
the Gutenberg end marker. Bracketed illustrations and the Alaska glossary
were removed. Wrapped lines were joined within paragraphs. Capitalization,
punctuation, chapter headings, and paragraph breaks were generally preserved.
This is heuristic cleaning, not a manual correction of every transcription.

## Split and overlap policy

The first book contained 325,906 cleaned characters. Its first 293,315
characters formed the original training partition; the final 32,591 formed
validation. This exact validation text was kept when adding the other books.
Each added paragraph sharing an exact, case-sensitive sequence of 20
whitespace-delimited words with validation was removed. This does not detect
all shorter quotations, paraphrases, or topical similarity.

The final training file has 1,286,809 characters (224,757 whitespace-delimited
words); validation has 32,591 characters (5,656 such words). The sorted union
contains 94 characters. Validation therefore informs the character inventory,
but does not contribute to gradient updates or baseline frequency counts.
Validation was used repeatedly for model selection; there is no untouched test
set. These are development results, not unbiased final test results.

The checked-in cleaned files are the authoritative inputs for reproduction.
SHA-256 hashes are in ../docs/baselines.json. Byte counts differ from character
counts because the files contain UTF-8 punctuation and accented characters.
