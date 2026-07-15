"""Evaluation loop and metrics logging for Cerebro.

Provides:
- Periodic evaluation during training
- Loss and perplexity tracking
- WandB and TensorBoard integration
- Benchmark suite (MMLU, HumanEval, HellaSwag stubs)
"""

from __future__ import annotations

import os
import time
import json
import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("cerebro.evaluator")


@dataclass
class EvalResult:
    """Result from an evaluation run."""
    loss: float
    perplexity: float
    tokens_evaluated: int
    elapsed_seconds: float
    tokens_per_second: float
    metadata: dict = field(default_factory=dict)


class Evaluator:
    """Model evaluation engine.

    Runs evaluation on a held-out dataset, computing loss and perplexity.
    Supports streaming results to WandB and TensorBoard.

    Args:
        model: Cerebro model instance.
        config: Model configuration.
        device: Target device.
        eval_interval: Steps between evaluations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config=None,
        device: str = "auto",
        eval_interval: int = 500,
    ) -> None:
        self.model = model
        self.config = config
        self.eval_interval = eval_interval

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.history: list[EvalResult] = []

    @torch.no_grad()
    def evaluate(self, eval_dataloader) -> EvalResult:
        """Run evaluation on a dataset.

        Args:
            eval_dataloader: DataLoader yielding {'input_ids', 'labels'} batches.

        Returns:
            EvalResult with loss, perplexity, and throughput.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(input_ids, labels=labels)
            loss = output["loss"]

            batch_tokens = (labels != 0).sum().item()  # exclude padding
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = min(float(torch.exp(torch.tensor(avg_loss)).item()), 1e6)

        result = EvalResult(
            loss=avg_loss,
            perplexity=perplexity,
            tokens_evaluated=total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=total_tokens / max(elapsed, 0.001),
        )

        self.history.append(result)
        self.model.train()
        return result

    def evaluate_perplexity(self, text: str, tokenizer) -> float:
        """Compute perplexity of a single text string.

        Args:
            text: Input text to evaluate.
            tokenizer: Cerebro tokenizer instance.

        Returns:
            Perplexity score.
        """
        self.model.eval()
        tokens = tokenizer.encode(text, add_bos=True)
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(self.device)
        labels = torch.tensor([tokens[1:]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, labels=labels)

        self.model.train()
        loss = output["loss"].item()
        return min(float(torch.exp(torch.tensor(loss)).item()), 1e6)


class MetricsLogger:
    """Unified metrics logging to WandB, TensorBoard, and JSON.

    Automatically detects and uses available logging backends.

    Args:
        project: Project name for WandB.
        run_name: Run name for identification.
        log_dir: Directory for TensorBoard logs and JSON metrics.
    """

    def __init__(
        self,
        project: str = "cerebro",
        run_name: str | None = None,
        log_dir: str = "logs",
    ) -> None:
        self.project = project
        self.run_name = run_name or f"run_{int(time.time())}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Detect available backends
        self.wandb = None
        self.tb_writer = None
        self._json_log: list[dict] = []

        self._init_wandb()
        self._init_tensorboard()

        if self.wandb is None and self.tb_writer is None:
            logger.info("MetricsLogger: No WandB or TensorBoard. Using JSON logging only.")

    def _init_wandb(self) -> None:
        """Initialize WandB if available."""
        try:
            import wandb
            self.wandb = wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config={},
            )
            logger.info("WandB initialized: %s/%s", self.project, self.run_name)
        except (ImportError, Exception):
            logger.debug("WandB not available", exc_info=True)
            self.wandb = None

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard initialized: %s", tb_dir)
        except ImportError:
            self.tb_writer = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to all available backends.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global step number.
        """
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self._json_log.append(entry)

        if self.wandb:
            self.wandb.log(metrics, step=step)

        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, global_step=step)

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_per_sec: float = 0.0,
    ) -> None:
        """Log a training step."""
        self.log({
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/tokens_per_sec": tokens_per_sec,
        }, step=step)

    def log_eval(self, step: int, result: EvalResult) -> None:
        """Log evaluation results."""
        self.log({
            "eval/loss": result.loss,
            "eval/perplexity": result.perplexity,
            "eval/tokens_evaluated": result.tokens_evaluated,
        }, step=step)

    def flush_json(self) -> None:
        """Flush JSON log to disk."""
        log_path = self.log_dir / "metrics.json"
        with open(log_path, "w") as f:
            json.dump(self._json_log, f, indent=2)

    def close(self) -> None:
        """Close all logging backends."""
        self.flush_json()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                logger.debug("WandB finish failed", exc_info=True)


class BenchmarkSuite:
    """Evaluation benchmark suite with real dataset loading.

    Provides evaluation harnesses for:
    - MMLU (Massive Multitask Language Understanding)
    - HumanEval (Code Generation)
    - HellaSwag (Commonsense Reasoning)
    - ARC (Abstract Reasoning Challenge)

    Supports loading from HuggingFace datasets hub or local JSONL files.
    """

    def __init__(self, model, tokenizer, device: str = "auto") -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    # ─── MMLU ───────────────────────────────────────────────────────

    def load_mmlu(self, subjects: list[str] | None = None) -> list[dict]:
        """Load MMLU dataset from HuggingFace or local JSONL.

        Args:
            subjects: Filter to specific subjects (e.g. ['college_physics', 'high_school_math']).
                      None loads all subjects.

        Returns:
            List of dicts with keys: question, subject, choices, answer.
        """
        samples = self._try_load_hf_dataset("cais/mmlu", "test", subjects)
        if samples:
            return samples
        return self._load_mmlu_jsonl(subjects)

    def _try_load_hf_dataset(
        self, dataset_name: str, split: str, subjects: list[str] | None = None
    ) -> list[dict]:
        """Attempt to load from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
            samples = []
            for item in ds:
                subject = item.get("subject", "")
                if subjects and subject not in subjects:
                    continue
                samples.append({
                    "question": item["question"],
                    "subject": subject,
                    "choices": [item.get(f"choice_{chr(65+i)}", item.get("choices", ["","","",""])[i])
                                for i in range(4)],
                    "answer": item.get("answer", "A"),
                })
            return samples
        except ImportError:
            return []

    def _load_mmlu_jsonl(self, subjects: list[str] | None = None) -> list[dict]:
        """Load MMLU from local JSONL data/mmlu/ directory."""
        import glob
        samples = []
        data_dir = Path("data/mmlu")
        if not data_dir.exists():
            return samples
        for filepath in glob.glob(str(data_dir / "*.jsonl")):
            subject = os.path.splitext(os.path.basename(filepath))[0]
            if subjects and subject not in subjects:
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj["subject"] = subject
                    samples.append(obj)
        return samples

    def run_mmlu_sample(self, question: str, choices: list[str]) -> str:
        """Evaluate a single MMLU-style multiple choice question.

        Uses log-probability scoring of the full answer choice text.

        Args:
            question: Question text.
            choices: List of answer choices (A, B, C, D).

        Returns:
            Predicted answer letter.
        """
        best_letter = "A"
        best_score = float("-inf")

        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            # Score full answer: "Question: ...\nA. choice_text\nAnswer: A"
            prompt = f"Question: {question}\n"
            for j, c in enumerate(choices):
                prompt += f"{chr(65+j)}. {c}\n"
            prompt += f"Answer: {letter}"

            tokens = self.tokenizer.encode(prompt, add_bos=True)
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(self.device)
            labels = torch.tensor([tokens[1:]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                self.model.eval()
                output = self.model(input_ids, labels=labels)
                score = -output["loss"].item()

            if score > best_score:
                best_score = score
                best_letter = letter

        return best_letter

    def evaluate_mmlu(
        self,
        subjects: list[str] | None = None,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Run full MMLU evaluation.

        Args:
            subjects: Subject filter.
            max_samples: Limit samples per subject for faster eval.

        Returns:
            Dict with per-subject and overall accuracy.
        """
        samples = self.load_mmlu(subjects)
        if not samples:
            return {"error": "No MMLU data found. Install datasets: pip install datasets"}

        if max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:max_samples]

        correct = 0
        by_subject: dict[str, tuple[int, int]] = {}

        for i, sample in enumerate(samples):
            predicted = self.run_mmlu_sample(sample["question"], sample["choices"])
            answer = sample["answer"]
            if predicted == answer:
                correct += 1
            subject = sample.get("subject", "unknown")
            prev = by_subject.get(subject, (0, 0))
            by_subject[subject] = (prev[0] + (1 if predicted == answer else 0), prev[1] + 1)

            if (i + 1) % 100 == 0:
                logger.info("MMLU: %d/%d acc=%.3f", i + 1, len(samples), correct / (i + 1))

        results = {"overall_accuracy": correct / len(samples), "num_samples": len(samples)}
        for subject, (sub_correct, sub_total) in sorted(by_subject.items()):
            results[f"mmlu/{subject}"] = sub_correct / sub_total

        return results

    # ─── HumanEval ──────────────────────────────────────────────────

    def load_humaneval(self) -> list[dict]:
        """Load HumanEval dataset from HuggingFace or local JSONL.

        Returns:
            List of dicts with keys: task_id, prompt, canonical_solution, test, entry_point.
        """
        samples = self._try_load_hf_dataset("openai/openai_humaneval", "test", None)
        if samples:
            return samples

        data_path = Path("data/humaneval/HumanEval.jsonl")
        if data_path.exists():
            samples = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            return samples
        return []

    def run_humaneval_sample(self, prompt: str, entry_point: str, test: str) -> dict:
        """Evaluate a single HumanEval coding task.

        Args:
            prompt: Function signature + docstring.
            entry_point: Function name to extract.
            test: Test code string.

        Returns:
            Dict with completion, passed status, and error if any.
        """
        full_prompt = f"""Write a Python function that solves the following problem.

{prompt}

```python
"""

        tokens = self.tokenizer.encode(full_prompt, add_bos=True)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            self.model.eval()
            generated = self.model.generate(
                input_ids, max_new_tokens=512, temperature=0.2, do_sample=True
            )

        output_tokens = generated[0].tolist()
        completion = self.tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

        # Extract the function code
        code = self._extract_code_block(completion)

        # Execute the test
        passed = False
        error = None
        try:
            namespace = {}
            exec(code, namespace)
            exec(test, namespace)
            namespace["check"](namespace[entry_point])
            passed = True
        except (AssertionError, TypeError, ValueError, NameError, SyntaxError, KeyError, ZeroDivisionError, IndexError, RuntimeError) as e:
            error = str(e)

        return {"completion": code, "passed": passed, "error": error}

    def _extract_code_block(self, text: str) -> str:
        """Extract code from text that may contain markdown fences."""
        import re
        # Try to find code between ```python and ```
        match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return everything after the first 'def' or 'import'
        lines = text.split("\n")
        code_lines = []
        started = False
        for line in lines:
            if not started and (line.strip().startswith("def ") or line.strip().startswith("import ")):
                started = True
            if started:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else text

    def evaluate_humaneval(
        self,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Run full HumanEval evaluation.

        Args:
            max_samples: Limit number of problems.

        Returns:
            Dict with pass@1 and detailed results.
        """
        samples = self.load_humaneval()
        if not samples:
            return {"error": "No HumanEval data found. Install datasets: pip install datasets"}

        if max_samples:
            samples = samples[:max_samples]

        passed = 0
        results = []

        for i, sample in enumerate(samples):
            result = self.run_humaneval_sample(
                sample["prompt"],
                sample.get("entry_point", "solution"),
                sample["test"],
            )
            results.append(result)
            if result["passed"]:
                passed += 1
            logger.info(
                "HumanEval: %d/%d (%s): %s",
                i + 1, len(samples), sample.get('task_id', f'task_{i}'),
                'PASS' if result['passed'] else 'FAIL',
            )

        return {
            "pass@1": passed / len(samples),
            "num_passed": passed,
            "num_total": len(samples),
        }

    # ─── HellaSwag ──────────────────────────────────────────────────

    def load_hellaswag(self) -> list[dict]:
        """Load HellaSwag dataset."""
        samples = self._try_load_hf_dataset("Rowan/hellaswag", "validation", None)
        if samples:
            return samples
        return []

    def run_hellaswag_sample(self, context: str, completions: list[str]) -> int:
        """Evaluate a single HellaSwag-style completion task.

        Args:
            context: Context text.
            completions: List of possible completions.

        Returns:
            Index of best completion.
        """
        best_idx = 0
        best_score = float("-inf")

        for i, completion in enumerate(completions):
            full_text = context + completion
            tokens = self.tokenizer.encode(full_text, add_bos=True)

            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(self.device)
            labels = torch.tensor([tokens[1:]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                self.model.eval()
                output = self.model(input_ids, labels=labels)
                score = -output["loss"].item()

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def evaluate_hellaswag(self, max_samples: int | None = None) -> dict[str, float]:
        """Run full HellaSwag evaluation."""
        samples = self.load_hellaswag()
        if not samples:
            return {"error": "No HellaSwag data found. Install datasets: pip install datasets"}

        if max_samples:
            samples = samples[:max_samples]

        correct = 0
        for i, sample in enumerate(samples):
            predicted = self.run_hellaswag_sample(
                sample.get("ctx", ""),
                sample.get("endings", []),
            )
            if predicted == int(sample.get("label", -1)):
                correct += 1
            if (i + 1) % 100 == 0:
                logger.info("HellaSwag: %d/%d acc=%.3f", i + 1, len(samples), correct / (i + 1))

        return {"accuracy": correct / len(samples), "num_samples": len(samples)}

    # ─── Full Evaluation ───────────────────────────────────────────

    def run_all_benchmarks(
        self,
        mmlu_subjects: list[str] | None = None,
        max_samples: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run all available benchmarks.

        Args:
            mmlu_subjects: MMLU subject filter.
            max_samples: Limit samples for quick evaluation.

        Returns:
            Dict of benchmark_name -> results.
        """
        results = {}

        logger.info("Running MMLU benchmark...")
        results["mmlu"] = self.evaluate_mmlu(subjects=mmlu_subjects, max_samples=max_samples)

        logger.info("Running HumanEval benchmark...")
        results["humaneval"] = self.evaluate_humaneval(max_samples=max_samples)

        logger.info("Running HellaSwag benchmark...")
        results["hellaswag"] = self.evaluate_hellaswag(max_samples=max_samples)

        return results
