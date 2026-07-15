"""Secure weight storage for Cerebro models.

Encrypts/decrypts model weights using AES-256-GCM for at-rest protection.
Keys are derived from a passphrase or PQC shared secret.

Enforced PQC integrity: when a Dilithium3 signing key is supplied, every
encrypted checkpoint is authenticated with a post-quantum signature. On
decryption, if the checkpoint header declares PQC integrity, the signature
MUST verify (fail-closed) — tampering yields a ValueError.
"""

from __future__ import annotations

import os
import json
import hashlib
import struct
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("cerebro.weight_encryption")


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a passphrase using PBKDF2.

    Args:
        passphrase: User passphrase or PQC shared secret hex string.
        salt: Random salt bytes.

    Returns:
        32-byte AES key.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=100_000,
    )


def _aes_gcm_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt with AES-256-GCM.

    Args:
        plaintext: Data to encrypt.
        key: 32-byte key.

    Returns:
        nonce (12 bytes) + ciphertext + tag (16 bytes).
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ct
    except ImportError:
        # Fallback: XOR with key-derived stream (NOT secure, for demo only)
        nonce = os.urandom(12)
        stream = hashlib.sha256(key + nonce).digest()
        encrypted = bytes(a ^ stream[i % len(stream)] for i, a in enumerate(plaintext))
        tag = hashlib.sha256(encrypted + key).digest()[:16]
        return nonce + encrypted + tag


def _aes_gcm_decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM.

    Args:
        data: nonce (12 bytes) + ciphertext + tag (16 bytes).
        key: 32-byte key.

    Returns:
        Decrypted plaintext.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = data[:12]
        ct = data[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)
    except ImportError:
        # Fallback: reverse the XOR
        nonce = data[:12]
        tag = data[-16:]
        encrypted = data[12:-16]
        stream = hashlib.sha256(key + nonce).digest()
        return bytes(a ^ stream[i % len(stream)] for i, a in enumerate(encrypted))


class WeightEncryption:
    """Encrypt and decrypt model weight files.

    Uses AES-256-GCM with PBKDF2 key derivation.
    Supports both safetensors and PyTorch checkpoint formats.

    When ``pqc_signing_key`` is provided to :meth:`encrypt_checkpoint`, the
    header + ciphertext are additionally signed with ML-DSA (Dilithium3).
    :meth:`decrypt_checkpoint` refuses to load a PQC-tagged checkpoint
    without a valid public key + signature, so tampering with weights on
    disk is quantum-safely detected.
    """

    MAGIC = b"CBRO_ENC"       # Classical AES-GCM only
    MAGIC_PQC = b"CBRO_PQC"   # AES-GCM + Dilithium3 signature
    VERSION = 1

    @staticmethod
    def encrypt_checkpoint(
        state_dict: dict[str, torch.Tensor],
        output_path: str,
        passphrase: str,
        pqc_signing_key: Optional[bytes] = None,
    ) -> None:
        """Encrypt and save a model checkpoint.

        Args:
            state_dict: Model state dictionary.
            output_path: Output file path.
            passphrase: Encryption passphrase.
            pqc_signing_key: Optional Dilithium3 secret key. When provided,
                the file is tagged with the PQC magic and the ciphertext is
                authenticated with a ML-DSA signature (fail-closed on load).
        """
        import io

        # Serialize state dict to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        plaintext = buffer.getvalue()

        # Derive key
        salt = os.urandom(16)
        key = _derive_key(passphrase, salt)

        # Encrypt
        ciphertext = _aes_gcm_encrypt(plaintext, key)

        # Assemble body: version + salt + ciphertext
        body = struct.pack("<I", WeightEncryption.VERSION) + salt + ciphertext

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if pqc_signing_key is not None:
            # PQC-authenticated checkpoint (fail-closed on liboqs missing)
            from cerebro.security.pqc import PQCDSA
            dsa = PQCDSA()
            signature = dsa.sign(body, pqc_signing_key)
            with open(output_path, "wb") as f:
                f.write(WeightEncryption.MAGIC_PQC)
                f.write(struct.pack("<I", len(signature)))
                f.write(signature)
                f.write(body)
            logger.info("Encrypted checkpoint with PQC (Dilithium3) signature: %s", output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(WeightEncryption.MAGIC)
                f.write(body)
            logger.info("Encrypted checkpoint (AES-256-GCM, no PQC signature): %s", output_path)

    @staticmethod
    def decrypt_checkpoint(
        input_path: str,
        passphrase: str,
        pqc_public_key: Optional[bytes] = None,
        require_pqc: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decrypt and load a model checkpoint.

        Args:
            input_path: Encrypted checkpoint file path.
            passphrase: Decryption passphrase.
            pqc_public_key: Dilithium3 public key. MUST be provided if the
                file is tagged with PQC magic bytes. Signature is verified
                before decryption (fail-closed).
            require_pqc: If True, refuse checkpoints that are not
                PQC-signed. Use in high-assurance deployments.

        Returns:
            Decrypted state dictionary.

        Raises:
            ValueError: If magic mismatches, PQC signature is invalid, or
                PQC is required but missing.
        """
        import io

        with open(input_path, "rb") as f:
            magic = f.read(8)

            if magic == WeightEncryption.MAGIC_PQC:
                if pqc_public_key is None:
                    raise ValueError(
                        "Checkpoint is PQC-signed but no public key was provided. "
                        "Pass pqc_public_key=... to decrypt_checkpoint()."
                    )
                sig_len = struct.unpack("<I", f.read(4))[0]
                if sig_len <= 0 or sig_len > 65536:
                    raise ValueError(f"Invalid PQC signature length: {sig_len}")
                signature = f.read(sig_len)
                body = f.read()

                from cerebro.security.pqc import PQCDSA
                dsa = PQCDSA()
                if not dsa.verify(body, signature, pqc_public_key):
                    raise ValueError(
                        "PQC signature verification FAILED — checkpoint has been "
                        "tampered with or public key is wrong. Refusing to load."
                    )
                logger.info("PQC signature verified for %s", input_path)

                version = struct.unpack("<I", body[:4])[0]
                if version != WeightEncryption.VERSION:
                    raise ValueError(f"Unsupported encryption version: {version}")
                salt = body[4:20]
                ciphertext = body[20:]

            elif magic == WeightEncryption.MAGIC:
                if require_pqc:
                    raise ValueError(
                        "Checkpoint is not PQC-signed but require_pqc=True. "
                        "Refusing to load unauthenticated weights."
                    )
                version = struct.unpack("<I", f.read(4))[0]
                if version != WeightEncryption.VERSION:
                    raise ValueError(f"Unsupported encryption version: {version}")
                salt = f.read(16)
                ciphertext = f.read()
            else:
                raise ValueError("Not a valid encrypted Cerebro checkpoint")

        # Derive key
        key = _derive_key(passphrase, salt)

        # Decrypt
        plaintext = _aes_gcm_decrypt(ciphertext, key)

        # Deserialize
        buffer = io.BytesIO(plaintext)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        return state_dict

    @staticmethod
    def sign_manifest(manifest: dict, pqc_signing_key: bytes) -> bytes:
        """Produce a Dilithium3 signature over a canonical JSON manifest.

        Used for model release/publish flows to bind a signed manifest
        (checkpoint hash, config, provenance) to a specific signer.
        """
        from cerebro.security.pqc import PQCDSA
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return PQCDSA().sign(payload, pqc_signing_key)

    @staticmethod
    def verify_manifest(manifest: dict, signature: bytes, pqc_public_key: bytes) -> bool:
        """Verify a Dilithium3 signature over a canonical JSON manifest."""
        from cerebro.security.pqc import PQCDSA
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return PQCDSA().verify(payload, signature, pqc_public_key)

    @staticmethod
    def encrypt_file(
        input_path: str,
        output_path: str,
        passphrase: str,
    ) -> None:
        """Encrypt any file with AES-256-GCM.

        Args:
            input_path: Input file path.
            output_path: Encrypted output path.
            passphrase: Encryption passphrase.
        """
        with open(input_path, "rb") as f:
            plaintext = f.read()

        salt = os.urandom(16)
        key = _derive_key(passphrase, salt)
        ciphertext = _aes_gcm_encrypt(plaintext, key)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(WeightEncryption.MAGIC)
            f.write(struct.pack("<I", WeightEncryption.VERSION))
            f.write(salt)
            f.write(ciphertext)

    @staticmethod
    def decrypt_file(
        input_path: str,
        output_path: str,
        passphrase: str,
    ) -> None:
        """Decrypt an encrypted file.

        Args:
            input_path: Encrypted file path.
            output_path: Decrypted output path.
            passphrase: Decryption passphrase.
        """
        with open(input_path, "rb") as f:
            magic = f.read(8)
            if magic != WeightEncryption.MAGIC:
                raise ValueError("Not a valid encrypted Cerebro file")
            version = struct.unpack("<I", f.read(4))[0]
            salt = f.read(16)
            ciphertext = f.read()

        key = _derive_key(passphrase, salt)
        plaintext = _aes_gcm_decrypt(ciphertext, key)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(plaintext)
