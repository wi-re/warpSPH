# Adding a paper

Drop the PDF in `literature/` and ask for a sync. That is genuinely all the
manual work there is — but the steps below are what a sync has to *do*, and
they are written down because most of them exist to catch a specific mistake
that has already happened here at least once.

> **Never commit the PDF.** `.gitignore` covers `literature/*` (except `*.md`
> and `*.bib`) and `*.pdf` repo-wide. The documents are third-party copyrighted
> material; only the metadata is tracked. If `git status` ever shows a `.pdf`,
> stop and fix the ignore rule rather than committing it.

## The one-line version

Copy the PDF in under any name, then:

> Sync `literature/`: reconcile the PDFs against the manifest, verify the
> BibTeX and the abstracts against the documents, and rename anything that does
> not follow `<bibkey>_<slug>.pdf`.

Finish by running the checker, which fails if any of it went stale:

```bash
python scripts/check_literature.py
```

## What the sync has to do

The lookup mechanics — which API, which endpoint, how to get an abstract out of
a two-column PDF — are in the **`paper-lookup` skill**
(`.claude/skills/paper-lookup/SKILL.md`). This file is the repo-specific
procedure that uses them.

### 1. Identify the paper from the document, not the filename

Read `pdfinfo` output and page 1. Do not name it from the filename it arrived
with, and do not name it from memory.

This is not a theoretical caution. Of the 36 papers here, the incoming
filenames were wrong about the **first author** once (`koschier18_viscosity.pdf`
is Weiler et al.), about the **year** twice and in the same direction
(`bender16_micropolar_sca.pdf` is SCA 2017; `bender17_micropolar.pdf` is TVCG
2019), about **publication status** once (`unpublished_analyticBoundaries.pdf`
carries the SPHERIC 2024 running header on every page), and carried no
information at all twice (`1-s2.0-S002199911200229X-main.pdf`).

Watch for the conference-paper / journal-extension pair specifically. They share
title, abstract and authors; the running header on page 2 is what separates
them. Five such pairs are already in this library: DFSPH, micropolar,
volume maps, multi-level memory, and the analytic boundary integrals.

### 2. Take the fields from the DOI record

Get volume, issue, pages, article number and year from the DOI record, and
reconcile against the front matter. **Where they disagree, the DOI record wins**
and the disagreement goes in the entry's `note`.

The usual cause of disagreement is an author's-version PDF, whose ACM template
prints placeholders until typesetting fills them in. `winchenbach2021`'s copy
says `ACM Trans. Graph. 1, 1, Article 1 (January 2020)`; the published article
is `40(1), 2021`. If volume, issue and article number are all `1`, they are all
wrong.

### 3. Choose the key and the filename

Bib key is `firstauthor + year`, lowercase, no punctuation. On a collision,
suffix with a topic word — `band2018` / `band2018pb`, `bender2019vmaps` /
`bender2019micropolar` — never `a`/`b`, which nobody can decode later.

Filename is `<bibkey>_<slug>.pdf`, slug being a few hyphenated words from the
title. The checker enforces that the part before the first `_` is a key
`references.bib` defines, so filename and citation key stay in lockstep with no
lookup table.

### 4. Quote the abstract; never summarise it

Prefer the DOI record's stored abstract, fall back to the PDF text layer, and
record which in the `abstract from:` line. Keep the publisher's typos — the
SPHERIC 2023 abstract here says "GPU accelation" because the paper does.
Correcting them would quietly break the verbatim guarantee that makes
`ABSTRACTS.md` trustworthy.

Elsevier and IEEE mostly do not deposit abstracts, so those will come from the
PDF. That is fine; it is the two-column reassembly that needs care, and the
skill covers it.

**When the text layer itself is corrupt**, declare the defect rather than
quoting it or working around it. Some publisher PDFs interleave a floating
glyph — a vector overline, a subscript — into the middle of a word, or drop a
hyphen entirely, so the extraction reads `weaklycompressible`, or
`obtained by modi given by a Particle Shifting fying the pure ...`. Quoting
*that* puts gibberish in the file; quoting the sentence as it reads fails the
checker for a reason that has nothing to do with accuracy. Add one line per
defect to the block:

```
- **text-layer:** `weaklycompressible` -> `weakly compressible`
- **text-layer:** `by modi given by a particle shifting fying the` -> `by modifying the`
```

Each asserts "the text layer says the left side where the document says the
right side". The checker applies them to the extracted text before matching, so
the verbatim check still runs at full strictness — against a transformation
that is written down and reviewable. Keep each repair to the smallest span that
fixes the artifact; the checker rejects any repair over
`MAX_REPAIR_WORDS` words, so this cannot quietly become "and here is the
abstract I wanted". `antuono2021` is the worked example.

### 5. Update all four files together

| file | what changes |
|---|---|
| `MANIFEST.md` | a table row in the right section, and the count in "What is here" |
| `references.bib` | the entry, in the matching section |
| `ABSTRACTS.md` | the block: key, file, title, authors, venue, DOI, relevance, abstract |
| `DFSPH_IMPROVEMENT_PLAN.md` | only if the paper closes or reopens a §5 question |

The `relevance` line in `ABSTRACTS.md` and the `what it is` column in
`MANIFEST.md` are this repository's editorial notes — say what the paper does
*for this codebase*, not what it is about in general. "MLS pressure boundaries"
is the title; "recomputes boundary pressure inside the solver iteration, which
this codebase does not" is the relevance.

### 6. Run the checker

```bash
python scripts/check_literature.py
```

It reconciles three lists that drift independently — the PDFs on disk, the
`file` column of `MANIFEST.md`'s tables, and the keys in `references.bib` — and
then re-matches every PDF-sourced abstract word-for-word against its document.

That last check is the one worth having. A paraphrase reads perfectly well and
is invisible on review; the first pass at the DFSPH abstract here was a
transcription that had acquired three sentences the paper does not contain, and
only the mechanical check caught it. Abstracts are allowed to break into a few
contiguous runs, because publishers interleave columns and inject copyright
blocks; they are not allowed to contain a word the document does not.

It needs `pdftotext` (poppler-utils). On a fresh clone with no PDFs it skips the
abstract check with a notice and still checks the manifest/bib correspondence.

## Removing or replacing a paper

Replacing a copy (author's version → published version, say): keep the key and
filename, re-verify the fields against the DOI record, re-extract the abstract,
re-run the checker. Note the swap in the entry's `note` if any field moved.

Removing one: delete its row, entry and block. If it is still referenced by
`DFSPH_IMPROVEMENT_PLAN.md`, move the entry to a `% --- Not obtained` section
at the end of `references.bib` rather than deleting it — the checker
understands that section and will not demand a PDF for the keys in it.
