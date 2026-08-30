"""Lean 4 powered mirror for the QuanQonscious hybrid simulator.

This module provides a proof-driven execution path that mirrors the
CUDA-Q centric workflow using the Lean 4 theorem prover.  It supports
serial, concurrent (async) and parallel (threaded) verification of the 29
Vedic sutras that underpin the system's reasoning fabric.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "Lean4SessionConfig",
    "Lean4MirrorResult",
    "Lean4Mirror",
    "VEDIC_SUTRAS",
]


VEDIC_SUTRAS: tuple[str, ...] = (
    # Primary sutras
    "Ekadhikena Purvena",
    "Nikhilam Navatashcaramam Dashatah",
    "Urdhva Tiryagbhyam",
    "Paravartya Yojayet",
    "Shunyam Saamyasamuccaye",
    "Anurupye Shunyamanyat",
    "Sankalana Vyavakalanabhyam",
    "Puranapuranabhyam",
    "Chalana-Kalanabhyam",
    "Yavadunam",
    "Vyashtisamanstih",
    "Sesanyankena Charamena",
    "Sopantyadvayamantyam",
    "Ekanyunena Purvena",
    "Gunitasamuccayah",
    "Gunakasamuccayah",
    # Upa-sutras
    "Anurupyena",
    "Sisyate Sesasamjnah",
    "Adyamadyenantyamantyena",
    "Kevalaih Saptakam Gunyat",
    "Vestanam",
    "Yavadunam Tavadunikrtya Varganca Yojayet",
    "Yavadunam Tavadunam",
    "Antyayor Dasake'pi",
    "Antyayoreva",
    "Samuccayagunitah",
    "Lopanasthapanabhyam",
    "Vilokanam",
    "Gunita Samuccayah Samuccaya Gunita",
)


@dataclass(frozen=True)
class Lean4SessionConfig:
    """Configuration parameters for :class:`Lean4Mirror`."""

    # No imports by default. The statements this mirror renders
    # (``vedic.external.lean_props``) are core-Lean ``Int`` arithmetic, so a
    # bare toolchain compiles them. The previous default was ``("Mathlib",)``,
    # which nothing in the emitted body used and which made every generated
    # script unverifiable without a Mathlib build -- forcing the end-to-end
    # test to skip and the compile test to strip the imports and substitute a
    # body that needed nothing. A caller whose statements genuinely need
    # Mathlib passes imports=("Mathlib",) explicitly.
    imports: Sequence[str] = ()
    # ``open scoped BigOperators`` is Mathlib syntax and was unused; the two
    # set_options are core Lean and are kept, since a large emitted
    # conjunction is exactly what exhausts the default heartbeat budget.
    prelude: str = (
        "set_option maxHeartbeats 200000\n"
        "set_option maxRecDepth 512\n"
    )
    timeout: float = 60.0
    lean_path: str | None = None
    environment: Mapping[str, str] | None = None
    keep_artifacts: bool = False

    def resolved_lean_path(self) -> str:
        """Return the Lean executable path, raising when it is missing."""

        candidate = self.lean_path or shutil.which("lean")
        if candidate:
            return candidate
        raise FileNotFoundError(
            "Unable to locate the 'lean' executable. Install Lean 4 or set "
            "Lean4SessionConfig.lean_path explicitly."
        )


@dataclass(slots=True)
class Lean4MirrorResult:
    """Result of mirroring a single sutra execution in Lean 4."""

    sutra: str
    statement: str
    mode: str
    success: bool
    duration: float
    stdout: str
    stderr: str
    script_path: Path

    def as_dict(self) -> dict[str, object]:
        """Render the result in a serialisable mapping."""

        return {
            "sutra": self.sutra,
            "statement": self.statement,
            "mode": self.mode,
            "success": self.success,
            "duration": self.duration,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "script_path": str(self.script_path),
        }


class Lean4Mirror:
    """Lean 4 mirror that validates sutra-driven simulations."""

    #: The toolchain pin this package compiles against, committed at the
    #: package root. ``elan`` resolves a toolchain from the nearest
    #: ``lean-toolchain`` file walking up from the invocation directory, so a
    #: script written into a temp directory resolves nothing and every
    #: ``lean`` call fails with "no default toolchain configured" -- which is
    #: what happened here, unnoticed, because the only test that exercised
    #: ``run_serial`` was skipped for an unrelated reason (a Mathlib import
    #: the emitted body never used). The pin is copied into every artifact
    #: directory so the generated scripts compile from any working directory
    #: under exactly the toolchain the package declares.
    TOOLCHAIN_PIN = Path(__file__).resolve().parents[2] / "lean-toolchain"

    def __init__(self, config: Lean4SessionConfig | None = None):
        self.config = config or Lean4SessionConfig()
        self._artifact_root = Path(tempfile.mkdtemp(prefix="quanqonscious-lean4-"))
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._install_toolchain_pin()
        self._script_counter = itertools.count()

    def _install_toolchain_pin(self) -> None:
        """Copy the committed toolchain pin beside the generated scripts.

        Raises rather than proceeding without it: running under whichever
        toolchain elan happens to default to would make the verdict depend on
        the machine, and a Lean version mismatch is exactly the kind of
        difference that turns a real failure into a confusing pass.
        """
        if not self.TOOLCHAIN_PIN.is_file():
            raise FileNotFoundError(
                f"{self.TOOLCHAIN_PIN} is missing. It pins the Lean toolchain "
                f"this package's generated scripts are compiled under and is "
                f"tracked in git; restore it rather than letting elan pick a "
                f"default, which would make the mirror's verdict depend on "
                f"the machine.")
        pin = self.TOOLCHAIN_PIN.read_text(encoding="utf-8").strip()
        if not pin:
            raise ValueError(f"{self.TOOLCHAIN_PIN} is empty; it must name a toolchain")
        (self._artifact_root / "lean-toolchain").write_text(
            pin + "\n", encoding="utf-8")

    @property
    def _lean_path(self) -> str:
        """The Lean executable, resolved on first use rather than at __init__.

        Resolving eagerly made the whole object unconstructible without a Lean
        toolchain, including for *rendering* — which is pure string work and
        needs no compiler. Anything that actually runs Lean still gets the
        same FileNotFoundError, just at the point where it matters.
        """
        cached = getattr(self, "_lean_path_cache", None)
        if cached is None:
            cached = self.config.resolved_lean_path()
            self._lean_path_cache = cached
        return cached

    # ------------------------------------------------------------------
    # Script rendering utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return slug or "sutra"

    def _script_path_for(self, sutra: str) -> Path:
        slug = self._slugify(sutra)
        index = next(self._script_counter)
        return self._artifact_root / f"{slug}-{index}.lean"

    def _render_script(self, sutra: str, statement: str) -> str:
        imports = "\n".join(f"import {module}" for module in self.config.imports)
        prelude = textwrap.dedent(self.config.prelude).strip()
        header_lines = [imports]
        if prelude:
            header_lines.append("")
            header_lines.append(prelude)
        header_lines.extend(
            [
                "",
                f"/-!\nAuto-generated mirror script for the Vedic sutra: {sutra}.\n-/",
                f"def sutraStatement : Bool :=\n  {statement}",
                "",
                "def mirrorMain : IO Unit := do",
                "  if sutraStatement then",
                "    IO.println ""true""",
                "  else",
                '    throw <| IO.userError "mirror validation failed"',
                "",
                "#eval mirrorMain",
            ]
        )
        return "\n".join(header_lines)

    def _write_script(self, sutra: str, statement: str) -> Path:
        path = self._script_path_for(sutra)
        script = self._render_script(sutra, statement)
        path.write_text(script, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Core execution primitives
    # ------------------------------------------------------------------
    def _invoke_lean(self, script_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._lean_path, script_path.name],
            cwd=str(script_path.parent),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **(self.config.environment or {})},
            timeout=self.config.timeout,
        )

    def _interpret_result(
        self,
        sutra: str,
        statement: str,
        mode: str,
        process: subprocess.CompletedProcess[str],
        start_time: float,
        script_path: Path,
    ) -> Lean4MirrorResult:
        success = process.returncode == 0 and "true" in process.stdout.splitlines()[-1:]
        duration = time.perf_counter() - start_time
        return Lean4MirrorResult(
            sutra=sutra,
            statement=statement,
            mode=mode,
            success=success,
            duration=duration,
            stdout=process.stdout,
            stderr=process.stderr,
            script_path=script_path,
        )

    def _clean_artifact(self, path: Path) -> None:
        if self.config.keep_artifacts:
            return
        # missing_ok states the postcondition -- "this path is gone" -- rather
        # than catching the miss, so no other error can be absorbed with it.
        path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def run_serial(self, sutra_statements: Mapping[str, str]) -> list[Lean4MirrorResult]:
        results: list[Lean4MirrorResult] = []
        for sutra, statement in sutra_statements.items():
            script_path = self._write_script(sutra, statement)
            start = time.perf_counter()
            process = self._invoke_lean(script_path)
            result = self._interpret_result(
                sutra=sutra,
                statement=statement,
                mode="serial",
                process=process,
                start_time=start,
                script_path=script_path,
            )
            results.append(result)
            self._clean_artifact(script_path)
        return results

    async def run_concurrent(self, sutra_statements: Mapping[str, str]) -> list[Lean4MirrorResult]:
        async def _run_single(sutra: str, statement: str) -> Lean4MirrorResult:
            script_path = self._write_script(sutra, statement)
            start = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                self._lean_path,
                script_path.name,
                cwd=str(script_path.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.config.environment or {})},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout
                )
            finally:
                self._clean_artifact(script_path)
            completed = subprocess.CompletedProcess(
                args=process.args,
                returncode=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )
            return self._interpret_result(
                sutra=sutra,
                statement=statement,
                mode="concurrent",
                process=completed,
                start_time=start,
                script_path=script_path,
            )

        tasks = [
            asyncio.create_task(_run_single(sutra, statement))
            for sutra, statement in sutra_statements.items()
        ]
        return await asyncio.gather(*tasks)

    def run_parallel(
        self,
        sutra_statements: Mapping[str, str],
        max_workers: int | None = None,
    ) -> list[Lean4MirrorResult]:
        def _worker(args: tuple[str, str]) -> Lean4MirrorResult:
            sutra, statement = args
            script_path = self._write_script(sutra, statement)
            start = time.perf_counter()
            process = self._invoke_lean(script_path)
            result = self._interpret_result(
                sutra=sutra,
                statement=statement,
                mode="parallel",
                process=process,
                start_time=start,
                script_path=script_path,
            )
            self._clean_artifact(script_path)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_worker, sutra_statements.items()))

    # Context manager support to dispose of artefacts eagerly
    def close(self) -> None:
        if self.config.keep_artifacts:
            return
        try:
            shutil.rmtree(self._artifact_root)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "Lean4Mirror":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - resource cleanup
        self.close()

