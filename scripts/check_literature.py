#!/usr/bin/env python3
"""Reconcile literature/ against its metadata, and re-verify every abstract.

Three checks, no network:

  1. The PDFs on disk, the `file` column of MANIFEST.md's table, and the keys in
     references.bib all name the same set of papers.
  2. Filenames follow `<bibkey>_<slug>.pdf`.
  3. Every abstract quoted in ABSTRACTS.md or ABSTRACTS_EXTENDED.md that claims a
     PDF source still appears verbatim in that PDF's text layer.

Check 3 is the one worth having. An abstract is easy to paraphrase by accident,
and a paraphrase that reads well is invisible on review -- so the text is
matched word-for-word rather than eyeballed. It is allowed to break into a few
contiguous runs, because several publishers interleave the two abstract columns
in reading order or drop a copyright block into the middle of it; what is not
allowed is a word that appears in ABSTRACTS.md and nowhere in the document.

Abstracts taken from a DOI record instead of the page are checked loosely (word
overlap), since the publisher's stored abstract and the typeset page legitimately
differ in ligatures, dashes and the odd corrected typo. ABSTRACTS_EXTENDED.md
holds the extended-set abstracts, most reconstructed from OpenAlex's inverted
index; all carry a non-PDF source, so they take the loose check.

A *corrupted* text layer is the third case, and it is not a paraphrase. Some
publisher PDFs interleave a floating glyph (a vector overline, a subscript)
into the middle of a word, or lose a hyphen entirely, so the extraction reads
`weaklycompressible` or `obtained by modi given by a Particle Shifting fying the
pure ...`. Quoting that verbatim would put gibberish in a file whose whole
purpose is to be trustworthy; quoting the sentence as it actually reads would
fail check 3 for a reason that has nothing to do with accuracy. So a block may
declare the defect:

    - **text-layer:** `weaklycompressible` -> `weakly compressible`
    - **text-layer:** `by modi given by a particle shifting fying the` -> `by modifying the`

Each line asserts "the document's text layer says the left-hand side where the
document itself says the right-hand side". The repairs are applied to the
*haystack* before matching, so check 3 still runs at full strictness against a
transformation that is written down, reviewable, and minimal -- rather than
being switched off. Both sides are normalised, so write them in whatever case
and punctuation reads clearest.

Requires `pdftotext` (poppler-utils) and the PDFs; skips check 3 with a notice
if either is absent, since the PDFs are deliberately not in the repository.

Usage:  python scripts/check_literature.py [--quiet]
Exits non-zero on any mismatch.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIT = os.path.join(ROOT, "literature")

MAX_RUNS = 4        # column seams / copyright blocks an abstract may be split by
MIN_RUN = 8         # words; a shorter "run" means the text was reworded
MIN_OVERLAP = 0.80  # for DOI-sourced abstracts
#: A declared text-layer repair may not rewrite more than this many words, so
#: the escape hatch cannot quietly become "and here is the abstract I wanted".
MAX_REPAIR_WORDS = 12


def read(name):
    return io.open(os.path.join(LIT, name), encoding="utf-8").read()


def norm(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def pdf_text(path):
    # first three pages: enough to clear a HAL/publisher cover sheet (leroy2014,
    # desbrun1996, keiser2006) and an abstract that wraps onto the next page.
    out = subprocess.run(["pdftotext", "-f", "1", "-l", "3", path, "-"],
                         capture_output=True, text=True).stdout
    return norm(out.replace("-\n", "").replace("\n", " "))


def cover(words, haystack):
    """Greedy longest-prefix cover of `words` by substrings of `haystack`.
    Returns the run lengths, or None if some word is absent entirely."""
    vocab = set(haystack.split())
    runs, i = [], 0
    while i < len(words):
        if words[i] not in vocab:
            return None
        lo, hi = 1, len(words) - i
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if " ".join(words[i:i + mid]) in haystack:
                lo = mid
            else:
                hi = mid - 1
        runs.append(lo)
        i += lo
    return runs


def main():
    quiet = "--quiet" in sys.argv
    problems = []

    def say(msg):
        if not quiet:
            print(msg)

    manifest, bib, abstracts = read("MANIFEST.md"), read("references.bib"), read("ABSTRACTS.md")
    ext_path = os.path.join(LIT, "ABSTRACTS_EXTENDED.md")
    abstracts_ext = read("ABSTRACTS_EXTENDED.md") if os.path.exists(ext_path) else ""

    on_disk = {f for f in os.listdir(LIT) if f.endswith(".pdf")}
    # only the table rows claim files; prose may name a superseded filename
    in_manifest = set()
    for line in manifest.splitlines():
        if line.startswith("|"):
            in_manifest.update(re.findall(r"`([A-Za-z0-9][^`]*\.pdf)`", line))
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    absent = (set(re.findall(r"^@\w+\{([^,]+),", bib[bib.index("% --- Not obtained"):], re.M))
              if "% --- Not obtained" in bib else set())
    have_keys = bib_keys - absent

    # 1. the three lists agree
    for f in sorted(in_manifest - on_disk):
        problems.append("MANIFEST names %s but it is not in literature/" % f)
    for f in sorted(on_disk - in_manifest):
        problems.append("%s is on disk but no MANIFEST row claims it" % f)

    # 2. filename convention, and the key it encodes exists in the bib
    for f in sorted(on_disk):
        if "_" not in f:
            problems.append("%s does not follow <bibkey>_<slug>.pdf" % f)
            continue
        key = f.split("_", 1)[0]
        if key not in have_keys:
            problems.append("%s encodes bib key '%s', which references.bib does not define" % (f, key))
    for key in sorted(have_keys):
        if not any(f.split("_", 1)[0] == key for f in on_disk):
            problems.append("references.bib defines '%s' outside the 'Not obtained' "
                            "section, but no PDF has that key" % key)

    say("%d PDFs, %d bib entries (%d of them not obtained)"
        % (len(on_disk), len(bib_keys), len(absent)))

    # 3. abstracts still match the documents
    block_re = r"### `([^`]+)`\n\n((?:- .*\n)+)\n((?:> .*\n)+)"
    blocks_core = re.findall(block_re, abstracts)
    blocks_ext = re.findall(block_re, abstracts_ext)
    blocks = blocks_core + blocks_ext
    say("%d abstract blocks (%d in ABSTRACTS.md, %d in ABSTRACTS_EXTENDED.md)"
        % (len(blocks), len(blocks_core), len(blocks_ext)))
    if not shutil.which("pdftotext"):
        say("note: pdftotext not found -- skipping the abstract check")
    elif not on_disk:
        say("note: no PDFs present -- skipping the abstract check")
    else:
        for key, meta, quote in blocks:
            fname = re.search(r"\*\*file:\*\* `([^`]+)`", meta).group(1)
            source = re.search(r"\*\*abstract from:\*\* (.+)", meta).group(1).strip()
            path = os.path.join(LIT, fname)
            if not os.path.exists(path):
                problems.append("%s: ABSTRACTS.md points at missing %s" % (key, fname))
                continue
            body = " ".join(line[2:] for line in quote.strip().split("\n"))
            words, hay = norm(body).split(), pdf_text(path)
            for bad, good in re.findall(
                    r"\*\*text-layer:\*\* `([^`]+)` -> `([^`]*)`", meta):
                bad, good = norm(bad), norm(good)
                if max(len(bad.split()), len(good.split())) > MAX_REPAIR_WORDS:
                    problems.append("%s: text-layer repair rewrites more than %d "
                                    "words -- too broad to be a glyph artifact"
                                    % (key, MAX_REPAIR_WORDS))
                elif bad not in hay:
                    problems.append("%s: declared text-layer defect %r is not in "
                                    "%s -- stale repair?" % (key, bad, fname))
                else:
                    hay = hay.replace(bad, good)
            if source.startswith("PDF"):
                runs = cover(words, hay)
                if runs is None:
                    problems.append("%s: abstract has wording absent from %s" % (key, fname))
                elif len(runs) > MAX_RUNS or min(runs) < MIN_RUN:
                    problems.append("%s: abstract matches %s only in %d fragments %s -- "
                                    "likely reworded, not quoted" % (key, fname, len(runs), runs))
                else:
                    say("  ok  %-28s verbatim in %s" % (key, runs))
            else:
                sh = [" ".join(words[i:i + 7]) for i in range(0, max(1, len(words) - 7), 6)]
                ratio = 1 - len([s for s in sh if s not in hay]) / len(sh)
                if ratio < MIN_OVERLAP:
                    problems.append("%s: DOI abstract overlaps %s by only %.0f%% -- "
                                    "is it the same paper?" % (key, fname, ratio * 100))
                else:
                    say("  ok  %-28s %.0f%% overlap (from %s)" % (key, ratio * 100, source))

    if problems:
        print("\n%d problem(s):" % len(problems))
        for p in problems:
            print("  - " + p)
        return 1
    print("\nliterature/ is consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
