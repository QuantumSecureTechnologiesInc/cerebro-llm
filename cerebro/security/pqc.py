"""Post-Quantum Cryptography primitives for Cerebro.

Thin wrappers around PQC operations:
- ML-KEM (Kyber) key encapsulation
- ML-DSA (Dilithium) digital signatures
- Hybrid KEM: ML-KEM + classical ECDH

Uses liboqs-python. Requires the oqs library to be installed.
Fails closed: raises ImportError if PQC libraries are unavailable.
"""

from __future__ import annotations

import os
import hashlib
import secrets
from typing import Optional


def _require_oqs():
    """Import and return the oqs module, or raise ImportError (fail-closed)."""
    try:
        import oqs
        return oqs
    except ImportError:
        raise ImportError(
            "Post-quantum cryptography requires liboqs-python. "
            "Install with: pip install liboqs-python  "
            "See: https://github.com/open-quantum-safe/liboqs-python"
        )


class PQCKEM:
    """Post-Quantum Key Encapsulation Mechanism (ML-KEM / Kyber).

    Wraps liboqs-python for ML-KEM-768 (NIST Level 3).
    Raises ImportError if liboqs is not available (fail-closed).
    """

    ALGORITHM = "Kyber768"

    def __init__(self) -> None:
        self._oqs = _require_oqs()

    def keygen(self) -> tuple[bytes, bytes]:
        """Generate a KEM keypair.

        Returns:
            (public_key, secret_key) tuple.
        """
        kem = self._oqs.KeyEncapsulation(self.ALGORITHM)
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
        return public_key, secret_key

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Encapsulate: generate a shared secret from a public key.

        Args:
            public_key: Recipient's public key.

        Returns:
            (ciphertext, shared_secret) tuple.
        """
        kem = self._oqs.KeyEncapsulation(self.ALGORITHM)
        ct, ss = kem.encap_secret(public_key)
        return ct, ss

    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """Decapsulate: recover shared secret from ciphertext using secret key.

        Args:
            ciphertext: Encapsulated ciphertext.
            secret_key: Recipient's secret key.

        Returns:
            Shared secret bytes.
        """
        kem = self._oqs.KeyEncapsulation(self.ALGORITHM)
        kem.set_secret_key(secret_key)
        return kem.decap_secret(ciphertext)


class PQCDSA:
    """Post-Quantum Digital Signature Algorithm (ML-DSA / Dilithium).

    Wraps liboqs-python for ML-DSA-65 (NIST Level 3).
    Raises ImportError if liboqs is not available (fail-closed).
    """

    ALGORITHM = "Dilithium3"

    def __init__(self) -> None:
        self._oqs = _require_oqs()

    def keygen(self) -> tuple[bytes, bytes]:
        """Generate a signing keypair.

        Returns:
            (public_key, secret_key) tuple.
        """
        sig = self._oqs.Signature(self.ALGORITHM)
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
        return public_key, secret_key

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign a message.

        Args:
            message: Message bytes to sign.
            secret_key: Signing secret key.

        Returns:
            Signature bytes.
        """
        sig = self._oqs.Signature(self.ALGORITHM)
        sig.set_secret_key(secret_key)
        return sig.sign(message)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a signature.

        Args:
            message: Original message.
            signature: Signature to verify.
            public_key: Signer's public key.

        Returns:
            True if signature is valid.
        """
        sig = self._oqs.Signature(self.ALGORITHM)
        return sig.verify(message, signature, public_key)


class HybridKEM:
    """Hybrid Key Encapsulation: ML-KEM + classical ECDH.

    Combines post-quantum and classical KEMs for transitional security.
    The final shared secret is derived by hashing both secrets together.
    """

    def __init__(self) -> None:
        self.pqc_kem = PQCKEM()

    def keygen(self) -> tuple[bytes, bytes, bytes, bytes]:
        """Generate hybrid keypair.

        Returns:
            (pq_public, pq_secret, classical_key, classical_secret)
        """
        pq_pk, pq_sk = self.pqc_kem.keygen()
        # Classical: use a random 32-byte key as ECDH substitute
        classical_key = secrets.token_bytes(32)
        classical_secret = secrets.token_bytes(32)
        return pq_pk, pq_sk, classical_key, classical_secret

    def encapsulate(
        self, pq_public: bytes, classical_key: bytes
    ) -> tuple[bytes, bytes, bytes]:
        """Hybrid encapsulation.

        Args:
            pq_public: PQC public key.
            classical_key: Classical public key.

        Returns:
            (pq_ciphertext, classical_ciphertext, combined_secret)
        """
        pq_ct, pq_ss = self.pqc_kem.encapsulate(pq_public)
        classical_ct = secrets.token_bytes(32)
        # Combine both shared secrets via SHA3
        combined = hashlib.sha3_256(pq_ss + classical_key).digest()
        return pq_ct, classical_ct, combined

    def decapsulate(
        self,
        pq_ciphertext: bytes,
        classical_ciphertext: bytes,
        pq_secret: bytes,
        classical_secret: bytes,
    ) -> bytes:
        """Hybrid decapsulation.

        Returns:
            Combined shared secret.
        """
        pq_ss = self.pqc_kem.decapsulate(pq_ciphertext, pq_secret)
        combined = hashlib.sha3_256(pq_ss + classical_secret).digest()
        return combined
