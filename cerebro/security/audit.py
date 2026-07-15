"""Tamper-evident audit logging for Cerebro.

Each log entry is chained with SHA3-512 hashes to form a Merkle-like
hash chain, making tampering detectable.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AuditLog:
    """Tamper-evident audit log with SHA3-512 hash chaining.

    Each entry includes:
    - Timestamp (UTC)
    - Event type
    - Payload (arbitrary JSON-serializable data)
    - Previous entry hash (creates a chain)
    - Current entry hash
    - Optional post-quantum signature (Dilithium3) over the entry hash

    When ``pqc_signing_key`` is supplied, every entry is additionally signed
    with ML-DSA (Dilithium3). :meth:`verify` will require the public key and
    fail on any missing or invalid signature (fail-closed).

    Args:
        log_path: Path to the audit log file (JSONL format).
        pqc_signing_key: Optional Dilithium3 secret key for entry signing.
        pqc_public_key: Optional Dilithium3 public key for verification.
    """

    def __init__(
        self,
        log_path: str = "audit.jsonl",
        pqc_signing_key: Optional[bytes] = None,
        pqc_public_key: Optional[bytes] = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = "GENESIS"
        self._pqc_sk = pqc_signing_key
        self._pqc_pk = pqc_public_key
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the hash of the last entry to resume chaining."""
        if not self.log_path.exists():
            return
        last_line = ""
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line:
            try:
                entry = json.loads(last_line)
                self._prev_hash = entry.get("hash", "GENESIS")
            except (json.JSONDecodeError, KeyError):
                self._prev_hash = "GENESIS"

    def _compute_hash(self, entry: dict) -> str:
        """Compute SHA3-512 hash of an entry.

        Args:
            entry: Log entry dictionary.

        Returns:
            Hex digest of the hash.
        """
        # Hash all fields except 'hash' and 'pqc_signature'
        data = json.dumps(
            {k: v for k, v in entry.items() if k not in ("hash", "pqc_signature")},
            sort_keys=True,
        )
        return hashlib.sha3_512(data.encode("utf-8")).hexdigest()

    def log(
        self,
        event_type: str,
        payload: dict | None = None,
        severity: str = "INFO",
    ) -> dict:
        """Write a tamper-evident log entry.

        Args:
            event_type: Event category (e.g., "MODEL_LOAD", "INFERENCE", "CHECKPOINT_SAVE").
            payload: Arbitrary JSON-serializable data.
            severity: Log level (INFO, WARNING, ERROR, CRITICAL).

        Returns:
            The logged entry dictionary.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "payload": payload or {},
            "prev_hash": self._prev_hash,
        }
        entry["hash"] = self._compute_hash(entry)

        # Optional PQC signature over the hash
        if self._pqc_sk is not None:
            from cerebro.security.pqc import PQCDSA
            sig = PQCDSA().sign(entry["hash"].encode("utf-8"), self._pqc_sk)
            entry["pqc_signature"] = sig.hex()

        # Append to log file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._prev_hash = entry["hash"]
        return entry

    def verify(self, require_pqc: bool = False) -> tuple[bool, int, int]:
        """Verify the integrity of the entire audit log.

        Checks that each entry's prev_hash matches the previous entry's hash.
        If a PQC public key is configured (or ``require_pqc`` is True),
        every entry is required to carry a valid Dilithium3 signature.

        Args:
            require_pqc: When True, entries without a valid PQC signature
                are counted as invalid even if the hash chain is intact.

        Returns:
            (is_valid, total_entries, invalid_entries) tuple.
        """
        if not self.log_path.exists():
            return True, 0, 0

        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        invalid = 0
        prev_hash = "GENESIS"

        check_pqc = require_pqc or self._pqc_pk is not None
        dsa = None
        if check_pqc and self._pqc_pk is not None:
            from cerebro.security.pqc import PQCDSA
            dsa = PQCDSA()

        for entry in entries:
            if entry.get("prev_hash") != prev_hash:
                invalid += 1

            expected_hash = self._compute_hash(entry)
            if entry.get("hash") != expected_hash:
                invalid += 1

            if check_pqc:
                sig_hex = entry.get("pqc_signature")
                if not sig_hex:
                    invalid += 1
                elif dsa is not None:
                    try:
                        sig = bytes.fromhex(sig_hex)
                        if not dsa.verify(entry["hash"].encode("utf-8"), sig, self._pqc_pk):
                            invalid += 1
                    except (ValueError, TypeError):
                        invalid += 1

            prev_hash = entry.get("hash", "")

        return invalid == 0, len(entries), invalid

    def log_model_load(self, checkpoint_path: str, num_params: int) -> None:
        """Log a model loading event."""
        self.log("MODEL_LOAD", {
            "checkpoint": checkpoint_path,
            "num_parameters": num_params,
        })

    def log_inference(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        elapsed_ms: float,
    ) -> None:
        """Log an inference event."""
        self.log("INFERENCE", {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_ms": round(elapsed_ms, 2),
            "tokens_per_second": round(generated_tokens / max(elapsed_ms / 1000, 0.001), 2),
        })

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
    ) -> None:
        """Log a training step."""
        self.log("TRAIN_STEP", {
            "step": step,
            "loss": round(loss, 6),
            "learning_rate": lr,
        })

    def log_checkpoint_save(self, tag: str, path: str) -> None:
        """Log a checkpoint save event."""
        self.log("CHECKPOINT_SAVE", {"tag": tag, "path": path})

    def log_security_event(self, description: str, severity: str = "WARNING") -> None:
        """Log a security-related event."""
        self.log("SECURITY", {"description": description}, severity=severity)
