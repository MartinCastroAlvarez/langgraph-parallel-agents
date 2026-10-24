# Bundled corpus

`corpus/` holds two public-domain works that the researcher agents search with BM25.
Both come from [Project Gutenberg](https://www.gutenberg.org/). The Project Gutenberg
header, footer and licence were removed, as that licence allows; they govern only the
Gutenberg trademark. Both works are in the public domain in the United States.

| File | Work | Gutenberg ebook |
| --- | --- | --- |
| `us-constitution.md` | The Constitution of the United States (1787), with Markdown headings per article and section | [#5](https://www.gutenberg.org/ebooks/5) |
| `art-of-war.md` | Sun Tzu, *The Art of War*, tr. Lionel Giles (1910), with one Markdown heading per chapter | [#17405](https://www.gutenberg.org/ebooks/17405) |

Passages are split along headings, and long sections into paragraph windows of at most
160 words. Ids are `<file>:<heading-slug>[-pN]`, e.g. `us-constitution:article-1-section-8`.
