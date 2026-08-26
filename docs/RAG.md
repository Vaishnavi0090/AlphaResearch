# Retrieval-Augmented Generation

Hybrid retrieval: BM25 fused with dense embeddings by Reciprocal Rank Fusion,
over a corpus whose every chunk carries enough provenance to audit a claim.

## Why both, and what each is for

The two halves fail in different ways, which is the whole argument for running
both:

| | Strength | Failure mode |
|---|---|---|
| BM25 | Exact terminology, numbers, tickers | Blind to paraphrase |
| Dense | Semantic similarity | Blurs precise distinctions |

**Fusion is by rank, not score.** BM25 scores are unbounded and
corpus-dependent; cosine similarity sits in [−1, 1]. Any fixed normalisation
between them is arbitrary and shifts as the corpus changes. Reciprocal Rank
Fusion uses only rank:

    score(d) = Σ_i  w_i / (k + rank_i(d))

with k = 60 from the original RRF paper. Scale-free and stable.

## Measured result: hybrid does not help on this corpus

Evaluated with 180 probes whose gold labels are derived from the corpus
generator, so the probe set is not tuned to any retriever:

| Mode | R@1 | MRR | nDCG@5 | latency |
|---|---|---|---|---|
| bm25 | **0.99** | **0.997** | 0.998 | 0.4 ms |
| dense (LSA) | 0.80 | 0.897 | 0.923 | 4.5 ms |
| hybrid | 0.99 | 0.994 | 0.995 | 4.7 ms |

Diagnosed, not hand-waved: the synthetic corpus is template-generated with only
~2,665 vocabulary terms across 800 chunks. The prose barely varies; only the
*numbers* differ. That is exactly BM25's strength and leaves nothing for
semantics to exploit.

**The machinery is not at fault.** On a paraphrase-rich corpus — documents
saying different things in different words — dense beats BM25 on top-1 accuracy
(50% vs 33%). "Why did profitability decline" shares no content word with
"gross margin contracted as input cost inflation outpaced pricing actions", and
only the dense path finds it. Asserted in
`test_dense_finds_paraphrase_that_bm25_misses`.

Hybrid stays the default because it ties BM25 here and wins on varied text,
making it the safer choice across corpus types. On real filings it is the right
default; on this one, BM25 alone would be equally good.

## Why LSA rather than a pretrained bi-encoder

`sentence-transformers` is supported and preferred when available. It is not the
default because model weights are unreachable in this environment
(huggingface.co returns HTTP 403), and a retrieval layer that silently fails
without network access is worse than one that is weaker but always works.

The fallback is genuine dense retrieval, not a stand-in: TF-IDF followed by
truncated SVD is Latent Semantic Analysis, where terms co-occurring across the
corpus collapse onto shared latent dimensions. Fitted on the corpus itself,
deterministic, ~64% variance retained at 128 dimensions.

Its honest limitation: semantics come only from co-occurrence *within the
indexed documents*, so it generalises less well than a pretrained model and
needs refitting when the corpus changes materially. `build_encoder("auto")`
tries sentence-transformers first, falls back to LSA, then to a plainly-named
`HashingEncoder` — and logs which one is active, so the encoder is never a
mystery. `/rag/diagnostics` reports it at runtime.

## Vector store

In-memory NumPy, deliberately not FAISS or a vector database. The corpus is
hundreds to low thousands of chunks; an exhaustive matrix multiply searches that
in under a millisecond and is *exact* where an ANN index is approximate. A
vector database would add a service dependency, a persistence format and an
approximation error in exchange for speed this workload does not need. The
interface (`add`, `search`, `save`, `load`) is narrow so swapping in FAISS later
is contained.

Metadata filtering happens **before** scoring. Post-filtering silently returns
fewer than `top_k`, which then looks like a relevance problem rather than a
filter problem.

## Provenance

Every chunk carries `section`, `page`, `filing_date`, `fiscal_period`, `ticker`,
`doc_type` and a unique `chunk_id`. "The 10-K says so" is not traceability if
you cannot say which part of which filing on which date.

Sections **carry forward across chunks**: a heading appears once, but the prose
beneath it spans many chunks. Labelling only the chunk containing the literal
"Item 1A" header left every following chunk — where the actual risk-factor text
lives — unattributed. Fixing this took section coverage from ~0 to 480/800
chunks.

## Original BM25 rationale (retained)

The lexical half is still doing most of the work here, for four reasons:

1. **Exact terminology matters.** "Goodwill impairment" and "asset impairment" are different accounting events. Embeddings blur precisely the distinctions financial analysis depends on.
2. **Numbers must match exactly.** A query for "3.2%" should find "3.2%", not "approximately 3%". The tokenizer preserves decimals, percentages and comma-separated figures.
3. **No model drift.** BM25 scoring is stable and inspectable; an embedding model upgrade silently changes every retrieval result.
4. **No dependency or inference cost.** No model download, no GPU, no API.

The counterpoint — lexical retrieval misses genuine paraphrase — is why the
dense half now exists.

## Pipeline

```
documents ─► metadata ─► sentence-aware chunking ─┬─► BM25 index
                          (section carry-forward) │
                                                  └─► encoder ─► vector store
                                                                      │
query ─┬─► tokenize ──► BM25 ranking ────────┐                        │
       └─► encode ────► cosine ranking ──────┴─► RRF fusion ◄─────────┘
                        (filtered first)            │
                                                    ▼
                                       coverage rerank → Citation objects
```

Set `RAGConfig.retriever` to `"bm25"`, `"dense"` or `"hybrid"` to isolate a
path — that is how the ablation above is measured.

## Chunking

Sentence-aware with overlap (default 900 chars, 150 overlap). Chunks break on sentence boundaries so a retrieved passage is readable, and overlap prevents a fact that straddles a boundary from becoming unretrievable.

Each chunk carries `doc_id`, `source`, `ticker`, `doc_type` and a unique `chunk_id` — the identifier the critic uses to verify a citation is real.

## BM25 scoring

$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)(k_1+1)}{f(t,d) + k_1(1 - b + b\frac{|d|}{\text{avgdl}})}$$

with $k_1 = 1.5$, $b = 0.75$. IDF is floored at zero — negative IDF for very common terms would let stopword-like tokens subtract from relevance.

Term-frequency saturation is the property that matters: a document repeating "goodwill" 50 times does not score 50× one that mentions it once. `test_bm25_saturates_term_frequency` asserts this.

Coverage reranking then promotes chunks matching *more distinct query terms* over chunks matching one term repeatedly.

## Two bugs found here

**1. The ticker filter was silently a no-op.** Metadata inference used `[A-Za-z]{1,6}`, which rejected alphanumeric tickers like `SY00`. Every chunk got `ticker=None`, so filtering by ticker matched nothing and was quietly ignored. Fixed, made strict (unknown-ticker chunks are excluded), and regression-tested by `test_infer_metadata_alphanumeric_ticker_not_dropped`.

**2. Trailing punctuation destroyed numeric recall.** The tokenizer allowed `.` inside tokens to preserve `3.2`, but sentence-final punctuation produced `2023.`, which never matched a query for `2023`. Since numeric facts are exactly what this retriever exists to find, this was quietly severe. Fixed by stripping trailing `.` and `-` while preserving internal ones.

Both were caught by tests, not by reading the code.

## Citations

Every result is a typed `Citation` with `doc_id`, `source`, `chunk_id`, `score` and `excerpt`. The critic validates every cited `chunk_id` against the index and removes any that does not resolve — so a fabricated citation cannot survive into a report.

Strict filtering has a visible consequence worth noting: querying a ticker with no documents returns **nothing** rather than loosely-related passages. That is the correct behaviour — empty evidence is honest, adjacent evidence is misleading.

## Corpus

`generate_corpus()` writes synthetic 10-Ks and earnings releases derived from the **actual synthetic fundamentals**, so retrieved claims are internally checkable against the same numbers the factor engine uses. Every document carries an explicit SYNTHETIC banner. Fiscal years are derived from the data rather than hardcoded, and `clean=True` clears stale files so a corpus from a different seed cannot mix in.

Full run: **320 documents → 800 chunks** across 40 tickers.

Real filings can be dropped into `RAGConfig.docs_dir` as `TICKER_10K_YEAR.txt` and are picked up automatically.
