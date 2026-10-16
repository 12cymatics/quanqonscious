"""SCAN, read from the copy vendored in this package, and the one text format.

Why the data is vendored
------------------------
``evaluate_scan`` used to call ``load_dataset("scan", split, split="test",
trust_remote_code=True)``. ``datasets`` 4.0 removed dataset loading scripts
and ``trust_remote_code`` with them, and ``scan`` on the Hub *is* a loading
script. Under the installed ``datasets`` 5.1 that call cannot load SCAN at all,
so no SCAN number could be produced. A benchmark whose loader depends on a
third party keeping a deprecated mechanism alive is not reproducible either.

The six task files of brendenlake/SCAN @ ``c4b756cb`` are vendored under
``data/scan/`` gzip-compressed (``gzip -9 -n``, so the archive bytes do not
depend on a timestamp), beside SCAN's BSD licence. ``SCAN_FILES`` pins the
SHA-256 and line count of each *decompressed* file, and ``load_scan`` checks
both on every read. A file that does not match is refused rather than
evaluated: a benchmark that is not the published one would still yield a
number, and nothing downstream could tell.

The text format, in one place
-----------------------------
The training corpus and the evaluator must agree on the prompt, the gold
continuation and the terminator. They used to agree by coincidence: the prompt
was a literal inside ``scan.py``, and no code converted SCAN into training
records, so the format the scorer needs was stated nowhere. ``prompt`` and
``record_text`` are now the only definitions, and both sides import them.

The terminator is ``"\\n"``, not the tokenizer's EOS. For SmolLM2 the pad
token *is* the EOS token, and ``DataCollatorForLanguageModeling`` sets the
label of every pad-id token to -100. So an EOS terminator is masked out of the
loss in every record, and a model trained on EOS-terminated text is never
taught where to stop. A newline survives as a label, and the evaluator scores
the first line.
"""
from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

#: Upstream commit the vendored files were taken from.
SCAN_COMMIT = "c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f"
SCAN_REPOSITORY = "https://github.com/brendenlake/SCAN"

DATA_DIR = Path(__file__).resolve().parent / "data" / "scan"

#: Scored answers end here. See the module docstring for why it is not EOS.
TERMINATOR = "\n"


@dataclass(frozen=True)
class ScanFile:
    """One vendored task file: where it lives upstream and what it must be."""
    upstream_path: str
    sha256: str          # of the decompressed text
    n_examples: int


#: (split, part) -> the vendored file. Hashes and counts are of the
#: decompressed text, which is byte-identical to the upstream file at
#: SCAN_COMMIT (checked against raw.githubusercontent.com when vendored).
SCAN_FILES: dict[tuple[str, str], ScanFile] = {
    ("simple", "train"): ScanFile(
        "simple_split/tasks_train_simple.txt",
        "941bb8a088c5f53ceff12fde902dc008933cf0c4203cc672c47b5f79d73262dd", 16728),
    ("simple", "test"): ScanFile(
        "simple_split/tasks_test_simple.txt",
        "1fe1c8f5a19d0dc40e41bab94278f45f66f3dc415a978bd29855a23440610fc6", 4182),
    ("length", "train"): ScanFile(
        "length_split/tasks_train_length.txt",
        "b034212dc7cf87ce62f3a067dd6ad49c4d5cd18d51d615d13825516bcb1ebbc2", 16990),
    ("length", "test"): ScanFile(
        "length_split/tasks_test_length.txt",
        "0760752ed34a75116ae11dcc1ee17526b7500060da1366540730fa24894d7db3", 3920),
    ("addprim_jump", "train"): ScanFile(
        "add_prim_split/tasks_train_addprim_jump.txt",
        "e3c78485692e028eed4205e1b8b3fd0e785ef2c12d45b5b89204dadf5b239e23", 14670),
    ("addprim_jump", "test"): ScanFile(
        "add_prim_split/tasks_test_addprim_jump.txt",
        "8147ae2651422c701162e174d9d66158daa57858a26b30317ac61797118ac7a5", 7706),
}


def vendored_path(split: str, part: str) -> Path:
    try:
        f = SCAN_FILES[(split, part)]
    except KeyError:
        raise ValueError(
            f"no SCAN file for split={split!r} part={part!r}; known: "
            f"{sorted(SCAN_FILES)}") from None
    return DATA_DIR / (Path(f.upstream_path).name + ".gz")


def read_verified(split: str, part: str) -> bytes:
    """The decompressed bytes of one task file, after checking them."""
    path = vendored_path(split, part)          # raises on an unknown pair
    f = SCAN_FILES[(split, part)]
    raw = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != f.sha256:
        raise ValueError(
            f"{path.name}: SHA-256 {digest} is not the pinned {f.sha256} of "
            f"{f.upstream_path} @ {SCAN_COMMIT[:8]}. This is not the published "
            f"benchmark, and it will not be evaluated as if it were.")
    return raw


def _parse_line(line: str, where: str) -> tuple[str, str]:
    if not line.startswith("IN: "):
        raise ValueError(f"{where}: does not start with 'IN: ': {line!r}")
    body = line[len("IN: "):]
    if body.count(" OUT: ") != 1:
        raise ValueError(f"{where}: needs exactly one ' OUT: ': {line!r}")
    command, actions = body.split(" OUT: ")
    if not command or not actions or command != command.strip() \
            or actions != actions.strip():
        raise ValueError(f"{where}: empty or padded field: {line!r}")
    return command, actions


def load_scan(split: str, part: str) -> list[tuple[str, str]]:
    """Every (command, actions) pair of one SCAN file, in file order.

    The whole file, always. There is no limit argument: a subset of a split
    is a different benchmark, and the count is checked against the pin.
    """
    text = read_verified(split, part).decode("utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"{split}/{part}: file does not end in a newline")
    lines = text[:-1].split("\n")
    pairs = [_parse_line(line, f"{split}/{part}:{i + 1}")
             for i, line in enumerate(lines)]
    expected = SCAN_FILES[(split, part)].n_examples
    if len(pairs) != expected:
        raise ValueError(
            f"{split}/{part}: {len(pairs)} examples, pinned {expected}")
    return pairs


def prompt(command: str) -> str:
    """What the model is shown at evaluation time."""
    return f"Command: {command}\nActions:"


def gold_continuation(actions: str) -> str:
    """What a correct model emits after ``prompt``: the answer, terminated."""
    return f" {actions}{TERMINATOR}"


def record_text(command: str, actions: str) -> str:
    """One training record: the evaluator's prompt followed by the gold
    continuation, so training and scoring see the same string."""
    return prompt(command) + gold_continuation(actions)
