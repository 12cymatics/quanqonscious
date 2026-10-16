"""First-line exact match: the verdict, the early stop, and the budget.

Why this exists
---------------
The SCAN scorer generated at most ``max_new_tokens=96``. The longest gold
answer of the length split is 264 SmolLM2 tokens, so exact match on that split
had a ceiling of exactly 0.0%. Nothing tested the ceiling, and a 0.0% would
have been recorded as a finding about the model.

The replacement computes the budget from the data and stops decoding once the
verdict is decided. Stopping early is only acceptable if it never changes a
verdict. ``test_stopping_early_never_changes_a_verdict`` checks that
exhaustively over every string up to length 6 on an alphabet chosen to hit
each branch of ``first_line_verdict``, and the scripted-model tests check it
through ``transformers``' real ``generate``.
"""
from __future__ import annotations

import itertools

import pytest
import torch
from transformers import DataCollatorForLanguageModeling, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from vedic.eval.first_line import (
    Example,
    continuation_length,
    first_line_verdict,
    generation_budget,
    score_first_lines,
)
from vedic.eval.scan import scan_examples
from vedic.eval.scan_data import record_text
from vedic.eval.scan_splits import SCAN_SPLITS

# ------------------------------------------------------------ the verdict


@pytest.mark.parametrize("text,terminated,verdict", [
    (" I_JUMP\n", False, True),            # the answer, terminated
    (" I_JUMP", True, True),               # the answer, then EOS
    (" I_JUMP  \n junk", False, True),     # trailing space; later lines ignored
    (" I_JUMP I_JUMP\n", False, False),    # too long
    (" I_JU\n", False, False),             # too short
    ("\n I_JUMP\n", False, False),         # a blank first line is the answer
    (" I_JU", False, None),                # a prefix of the answer: open
    (" I_JUMP", False, None),              # the answer, unterminated: open
    (" I_JUMP   ", False, None),           # the answer, then spaces: open
    ("   ", False, None),                  # only whitespace so far: open
    (" I_JUMPS", False, False),            # cannot be recovered
    (" I_LOOK", False, False),
    (" I_JUMP x", False, False),
    ("", True, False),                     # EOS at once: an empty answer
])
def test_the_verdict(text, terminated, verdict):
    assert first_line_verdict(text, "I_JUMP", terminated) is verdict


def _reference(full: str, target: str, terminated: bool) -> bool:
    """The verdict with no early stop: decode everything, then judge.
    Unterminated with no newline is no answer, and wrong."""
    return bool(first_line_verdict(full, target, terminated))


def _with_early_stop(full: str, target: str, terminated: bool) -> bool:
    """Judge at the first prefix where the verdict is decided."""
    for k in range(len(full) + 1):
        v = first_line_verdict(full[:k], target, terminated=False)
        if v is not None:
            return v
    return bool(first_line_verdict(full, target, terminated))


ALPHABET = ("a", "b", " ", "\t", "\n")


@pytest.mark.parametrize("target", ["a b", "ab", "b a b"])
def test_stopping_early_never_changes_a_verdict(target):
    """Every string over ALPHABET up to length 6, with and without EOS after it.

    Characters stand in for tokens: a token boundary is a subset of the
    character boundaries, and ``test_a_decided_verdict_never_changes`` shows
    that the first decided token boundary gives the same verdict as the first
    decided character.
    """
    n = 0
    for length in range(7):
        for chars in itertools.product(ALPHABET, repeat=length):
            full = "".join(chars)
            for terminated in (False, True):
                assert _with_early_stop(full, target, terminated) == \
                    _reference(full, target, terminated), (full, terminated)
                n += 1
    assert n == 2 * sum(len(ALPHABET) ** k for k in range(7))


@pytest.mark.parametrize("target", ["a b", "ab"])
def test_a_decided_verdict_never_changes(target):
    for length in range(7):
        for chars in itertools.product(ALPHABET, repeat=length):
            full = "".join(chars)
            decided = None
            for k in range(len(full) + 1):
                v = first_line_verdict(full[:k], target, terminated=False)
                if decided is not None:
                    assert v is decided, (full, k)
                elif v is not None:
                    decided = v
            if decided is not None:
                assert first_line_verdict(full, target, terminated=True) is decided


# -------------------------------------------- the real tokenizer and format

#: Longest gold continuation, terminator included, in SmolLM2 tokens. The same
#: on all three splits; the longest SCAN answer is 48 actions.
SCAN_BUDGET = 265


@pytest.mark.parametrize("split", SCAN_SPLITS)
def test_the_budget_is_the_longest_gold_continuation(smollm2_tokenizer, split):
    """Recomputed per example, so the prefix condition is checked on every one."""
    examples = scan_examples(split)
    lengths = [continuation_length(smollm2_tokenizer, e.prompt, e.gold_continuation)
               for e in examples]
    assert max(lengths) == SCAN_BUDGET
    assert generation_budget(smollm2_tokenizer, examples) == SCAN_BUDGET
    assert SCAN_BUDGET > 96, "the old fixed budget would have capped this split"


@pytest.mark.parametrize("split", SCAN_SPLITS)
def test_the_gold_continuation_decodes_to_the_answer(smollm2_tokenizer, split):
    """Tokenise each record, cut at the prompt, decode the rest: the scorer
    must read back exactly the target, so a perfect model scores 100%."""
    tok = smollm2_tokenizer
    for e in scan_examples(split):
        p = tok(e.prompt)["input_ids"]
        whole = tok(e.prompt + e.gold_continuation)["input_ids"]
        text = tok.decode(whole[len(p):], skip_special_tokens=True)
        assert first_line_verdict(text, e.target, terminated=False) is True


def test_the_terminator_is_trained_and_eos_would_not_be(smollm2_tokenizer):
    """Why records end in a newline. For SmolLM2 pad == eos, and the collator
    sets the label of every pad-id token to -100 -- including a real EOS."""
    tok = smollm2_tokenizer
    assert tok.pad_token_id == tok.eos_token_id
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    newline = tok(record_text("jump", "I_JUMP"))["input_ids"]
    eos = tok(record_text("jump", "I_JUMP")[:-1] + tok.eos_token)["input_ids"]
    b = collate([{"input_ids": newline}, {"input_ids": eos}])
    nl_id = tok("\n")["input_ids"][-1]
    assert newline[-1] == nl_id
    assert int(b["labels"][0, len(newline) - 1]) == nl_id
    assert eos[-1] == tok.eos_token_id
    assert int(b["labels"][1, len(eos) - 1]) == -100


# ----------------------------------------- scoring through the real generate


class Scripted(LlamaForCausalLM):
    """A model whose greedy continuation of each registered prompt is fixed.

    The logits put all mass on the scripted next token, then on EOS. Built on
    LlamaForCausalLM so ``generate`` -- stopping criteria, EOS handling,
    max_new_tokens -- is the real one. Counts forward calls, so an early stop
    is observable.
    """

    def __init__(self, vocab_size: int, eos_id: int) -> None:
        super().__init__(LlamaConfig(vocab_size=vocab_size, hidden_size=8,
                                     intermediate_size=8, num_hidden_layers=1,
                                     num_attention_heads=1, num_key_value_heads=1))
        self.generation_config.use_cache = False
        self.generation_config.eos_token_id = eos_id
        self.eos_id = eos_id
        self.scripts: dict[tuple[int, ...], list[int]] = {}
        self.calls = 0

    def forward(self, input_ids=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        ids = input_ids[0].tolist()
        for prompt, script in self.scripts.items():
            if tuple(ids[:len(prompt)]) == prompt:
                k = len(ids) - len(prompt)
                nxt = script[k] if k < len(script) else self.eos_id
                break
        else:
            raise AssertionError("unregistered prompt")
        logits = torch.full((1, len(ids), self.config.vocab_size), -1e9)
        logits[0, -1, nxt] = 0.0
        return CausalLMOutputWithPast(logits=logits)


def _scripted(tok, pairs: list[tuple[Example, str]]) -> Scripted:
    """Register each example's prompt with the text it should continue with."""
    m = Scripted(len(tok), tok.eos_token_id)
    for ex, continuation in pairs:
        p = tok(ex.prompt)["input_ids"]
        whole = tok(ex.prompt + continuation)["input_ids"]
        assert whole[:len(p)] == p
        m.scripts[tuple(p)] = whole[len(p):]
    return m


def test_scoring_through_generate(smollm2_tokenizer):
    tok = smollm2_tokenizer
    ex = [Example("Command: jump\nActions:", "I_JUMP"),
          Example("Command: walk\nActions:", "I_WALK"),
          Example("Command: look\nActions:", "I_LOOK"),
          Example("Command: run\nActions:", "I_RUN")]
    m = _scripted(tok, [
        (ex[0], " I_JUMP\n"),                    # right, newline
        (ex[1], " I_WALK"),                      # right, then EOS
        (ex[2], " I_LOOK I_LOOK\n"),             # one action too many
        (ex[3], " I_JUMP I_RUN\n"),              # wrong from the first token
    ])
    s = score_first_lines(m, tok, ex, device="cpu")
    assert (s.n_total, s.n_correct, s.n_budget_exhausted) == (4, 2, 0)


def test_a_wrong_answer_stops_at_the_token_that_decides_it(smollm2_tokenizer):
    """Exactly as many forward calls as tokens up to the deciding one.

    The gold answer is long on purpose. This test used a one-word answer, so
    the budget was 4 tokens and capped generation inside the bound by itself:
    with early stopping deleted it still passed (found by regeneration
    testing). The precondition below rules that out.
    """
    tok = smollm2_tokenizer
    target = " ".join(["I_RUN"] * 40)
    ex = [Example("Command: run forty times\nActions:", target)]
    m = _scripted(tok, [(ex[0], " I_JUMP" + " I_RUN" * 40 + "\n")])
    script = next(iter(m.scripts.values()))
    decide = next(k for k in range(1, len(script) + 1)
                  if first_line_verdict(tok.decode(script[:k], skip_special_tokens=True),
                                        target, terminated=False) is not None)
    s = score_first_lines(m, tok, ex, device="cpu")
    assert s.n_correct == 0
    assert decide < s.max_new_tokens, "vacuous: the budget alone would stop it here"
    assert m.calls == decide, f"{m.calls} forward calls; the verdict was decided at {decide}"


def test_the_longest_answer_fits_and_one_token_more_does_not(smollm2_tokenizer):
    """Both sides of the budget, on the example that sets it."""
    tok = smollm2_tokenizer
    examples = scan_examples("length")
    lengths = [continuation_length(tok, e.prompt, e.gold_continuation) for e in examples]
    longest = examples[lengths.index(max(lengths))]

    perfect = _scripted(tok, [(longest, longest.gold_continuation)])
    s = score_first_lines(perfect, tok, [longest], device="cpu")
    assert (s.n_correct, s.n_budget_exhausted, s.max_new_tokens) == (1, 0, SCAN_BUDGET)

    # The right answer with one extra space token before the newline: a
    # correct answer had it terminated, and it cannot within the budget.
    script = perfect.scripts[next(iter(perfect.scripts))]
    space = tok(" ")["input_ids"]
    assert len(space) == 1
    padded = Scripted(len(tok), tok.eos_token_id)
    padded.scripts = {k: script[:-1] + space + script[-1:] for k in perfect.scripts}
    assert len(next(iter(padded.scripts.values()))) == SCAN_BUDGET + 1
    assert tok.decode(next(iter(padded.scripts.values()))) == \
        " " + longest.target + " \n"
    s = score_first_lines(padded, tok, [longest], device="cpu")
    assert (s.n_correct, s.n_budget_exhausted) == (0, 1)
