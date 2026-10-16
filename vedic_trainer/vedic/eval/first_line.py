"""Greedy first-line exact match, with a budget computed from the data.

Shared by the SCAN and COGS evaluators, which differ only in their prompts.

What is scored
--------------
The model's answer is the text it generates before the first ``"\\n"`` (or
before EOS), with surrounding whitespace stripped. It is correct iff that
equals the gold target. A continuation that reaches the token budget without
terminating gives no answer and is scored wrong.

This differs in one deliberate way from the scorers it replaces, which
stripped the *whole* decode before taking the first line. That let an output
beginning with a blank line be scored on its second line. The answer is now
the first line, which is what the training format states.

The token budget is computed from the data
------------------------------------------
The SCAN scorer's ``max_new_tokens`` used to default to 96. The gold
continuation of the length-split test set runs to 264 SmolLM2 tokens, so 96
put a ceiling of exactly 0.0% on length-split exact match: no model, however
good, could have scored a single example, and a 0.0% would have read as a
finding. COGS had the same defect latently at 256. The budget is now the
longest gold continuation in the set (terminator included), counted in the
evaluated tokenizer's own tokens, and there is no parameter to lower it. A
model that emits the canonical tokenization of every correct answer can
score 100%.

Stopping once the verdict is decided
------------------------------------
Decoding stops as soon as the verdict can no longer change: at the first
newline, or once the text so far cannot be extended into the answer. That is
exact rather than an approximation: ``first_line_verdict`` returns a verdict
only when every possible continuation gives the same one, and
``vedic/eval/tests/test_first_line.py`` checks that stopping early never
changes a verdict. Without it, a model that never learned to stop would
decode every example to the full budget: about 14 hours per model on 4 cores
for the SCAN length split alone.

``n_budget_exhausted`` counts examples still undecided when the budget ran
out: answers that were a correct prefix and never terminated. They are the
*only* examples whose verdict the budget could have affected, so a result
with 0 there was not shaped by the budget at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

# One definition of the terminator and of a gold continuation, shared with the
# corpus builder, so training records and scoring cannot drift apart.
from vedic.eval.scan_data import TERMINATOR
from vedic.eval.scan_data import gold_continuation as _gold_continuation


def first_line_verdict(text: str, target: str, terminated: bool) -> bool | None:
    """The verdict on a continuation, or None while it can still go either way.

    ``text`` is the decoded continuation so far. ``terminated`` says the model
    emitted EOS, so nothing follows. ``target`` is the gold answer.
    """
    if TERMINATOR in text:
        return text.split(TERMINATOR, 1)[0].strip() == target
    if terminated:
        return text.strip() == target
    # No newline yet. The final answer is (text + more).strip() for some
    # newline-free `more`, and the leading part of that is text.lstrip() + more.
    so_far = text.lstrip()
    if target.startswith(so_far):
        return None                       # still a prefix of the answer
    if so_far.startswith(target) and so_far[len(target):].strip() == "":
        return None                       # the answer, then only whitespace
    return False                          # no continuation can recover it


class StopWhenDecided(StoppingCriteria):
    """Stop generating once ``first_line_verdict`` is no longer None."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_len: int,
                 target: str) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.target = target

    def __call__(self, input_ids: torch.LongTensor, scores: Any,
                 **kwargs: Any) -> torch.BoolTensor:
        if input_ids.shape[0] != 1:
            raise ValueError("first-line decoding is one example at a time")
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:],
                                     skip_special_tokens=True)
        decided = first_line_verdict(text, self.target, terminated=False) is not None
        return torch.full((1,), decided, dtype=torch.bool, device=input_ids.device)


def continuation_length(tokenizer: PreTrainedTokenizerBase, prompt: str,
                        continuation: str) -> int:
    """Tokens a model must generate after ``prompt`` to emit ``continuation``.

    Measured in context, as the difference between the tokenized whole and the
    tokenized prompt. Only defined when the prompt's tokens are an exact prefix
    of the whole's; otherwise "the tokens after the prompt" is not a
    well-defined quantity, and this raises rather than guess.
    """
    p = tokenizer(prompt)["input_ids"]
    whole = tokenizer(prompt + continuation)["input_ids"]
    if whole[:len(p)] != p:
        raise ValueError(
            f"the prompt's tokens are not a prefix of the record's for "
            f"{prompt!r}; the generation budget would be ill-defined")
    return len(whole) - len(p)


@dataclass(frozen=True)
class Example:
    prompt: str
    target: str

    @property
    def gold_continuation(self) -> str:
        return _gold_continuation(self.target)


@dataclass(frozen=True)
class FirstLineScore:
    n_total: int
    n_correct: int
    max_new_tokens: int
    n_budget_exhausted: int


def generation_budget(tokenizer: PreTrainedTokenizerBase,
                      examples: Sequence[Example]) -> int:
    """The longest gold continuation of the set, terminator included."""
    return max(continuation_length(tokenizer, e.prompt, e.gold_continuation)
               for e in examples)


def score_first_lines(model: nn.Module, tokenizer: PreTrainedTokenizerBase,
                      examples: Sequence[Example],
                      device: str | torch.device) -> FirstLineScore:
    """Greedy-decode every example and count exact first-line matches."""
    if not examples:
        raise ValueError("no examples to score")
    model.eval()
    eos_id = tokenizer.eos_token_id
    budget = generation_budget(tokenizer, examples)
    n_correct = 0
    n_exhausted = 0
    for ex in examples:
        inputs = tokenizer(ex.prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].size(1)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                num_beams=1,
                pad_token_id=eos_id,
                stopping_criteria=StoppingCriteriaList(
                    [StopWhenDecided(tokenizer, prompt_len, ex.target)]),
            )
        new_tokens = generated[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        ended_on_eos = new_tokens.numel() > 0 and int(new_tokens[-1]) == eos_id
        verdict = first_line_verdict(text, ex.target, terminated=ended_on_eos)
        if verdict is None:
            if new_tokens.numel() != budget:
                raise RuntimeError(
                    f"decoding stopped after {new_tokens.numel()} of {budget} "
                    f"tokens with the verdict still open on {ex.prompt!r}; "
                    f"nothing should stop it there")
            n_exhausted += 1              # open at the budget: no answer, wrong
        elif verdict:
            n_correct += 1
    return FirstLineScore(n_total=len(examples), n_correct=n_correct,
                          max_new_tokens=budget, n_budget_exhausted=n_exhausted)
