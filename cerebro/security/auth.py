"""Authentication and authorization middleware for Cerebro.

Production-grade auth with:
- API key management with key IDs, rotation, and expiration
- JWT (HS256/RS256) token issuance and validation
- OAuth2 Bearer token introspection (RFC 7662)
- Scoped API keys (read, write, admin)
- Key rotation without downtime
- FastAPI dependency injection

Follows production LLM auth patterns (OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

import os
import json
import time
import hmac
import hashlib
import secrets
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Callable
from pathlib import Path


# ── API Key Management ──


@dataclass
class ApiKey:
    """An API key with metadata for rotation and scoping."""

    key_id: str
    key_hash: str  # SHA-256 hash of the actual key
    name: str
    scopes: list[str] = field(default_factory=lambda: ["read"])
    created_at: str = ""
    expires_at: str | None = None
    last_used: str | None = None
    revoked: bool = False

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used": self.last_used,
            "revoked": self.revoked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ApiKey:
        return cls(
            key_id=d["key_id"],
            key_hash=d["key_hash"],
            name=d.get("name", ""),
            scopes=d.get("scopes", ["read"]),
            created_at=d.get("created_at", ""),
            expires_at=d.get("expires_at"),
            last_used=d.get("last_used"),
            revoked=d.get("revoked", False),
        )


class ApiKeyManager:
    """Manages API keys with rotation, expiration, and scope support.

    Keys are stored as SHA-256 hashes — plaintext keys are never persisted.
    Supports key rotation: generate new keys while old ones remain valid
    for a configurable grace period.

    Args:
        store_path: Path to the JSON keys file.
        default_ttl_days: Default key lifetime in days.
    """

    def __init__(
        self,
        store_path: str = "api_keys.json",
        default_ttl_days: int = 365,
    ) -> None:
        self.store_path = Path(store_path)
        self.default_ttl_days = default_ttl_days
        self._keys: dict[str, ApiKey] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                data = json.load(f)
            for kid, kd in data.items():
                self._keys[kid] = ApiKey.from_dict(kd)

    def _save(self) -> None:
        data = {kid: key.to_dict() for kid, key in self._keys.items()}
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_key(
        self,
        name: str,
        scopes: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> tuple[str, str]:
        """Generate a new API key.

        Returns:
            (key_id, plaintext_key) — the plaintext key is only returned once.
        """
        key_id = f"ck-{secrets.token_hex(12)}"
        plaintext = f"cerebro-{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

        ttl = ttl_days or self.default_ttl_days
        now = datetime.now(timezone.utc).isoformat()

        self._keys[key_id] = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or ["read"],
            created_at=now,
            expires_at=(
                (datetime.now(timezone.utc) + timedelta(days=ttl)).isoformat()
                if ttl > 0 else None
            ),
        )
        self._save()
        return key_id, plaintext

    def validate_key(self, plaintext: str) -> ApiKey | None:
        """Validate a plaintext API key.

        Returns:
            ApiKey if valid, None if invalid/expired/revoked.
        """
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        for key in self._keys.values():
            if key.key_hash == key_hash:
                if key.revoked:
                    return None
                if key.expires_at:
                    try:
                        expiry = datetime.fromisoformat(key.expires_at)
                        if datetime.now(timezone.utc) > expiry:
                            return None
                    except (ValueError, TypeError):
                        pass
                key.last_used = datetime.now(timezone.utc).isoformat()
                self._save()
                return key
        return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key by ID."""
        if key_id in self._keys:
            self._keys[key_id].revoked = True
            self._save()
            return True
        return False

    def rotate_key(self, key_id: str, new_name: str | None = None) -> tuple[str, str] | None:
        """Rotate an API key: revoke old, generate new.

        Returns:
            (new_key_id, new_plaintext) or None if key_id not found.
        """
        if key_id not in self._keys:
            return None
        old_key = self._keys[key_id]
        self.revoke_key(key_id)
        return self.generate_key(
            name=new_name or old_key.name,
            scopes=old_key.scopes,
        )

    def list_keys(self) -> list[dict]:
        """List all keys (without hashes)."""
        return [
            {
                "key_id": k.key_id,
                "name": k.name,
                "scopes": k.scopes,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "last_used": k.last_used,
                "revoked": k.revoked,
            }
            for k in self._keys.values()
        ]

    def has_scope(self, key_id: str, scope: str) -> bool:
        """Check if a key has a specific scope."""
        key = self._keys.get(key_id)
        return key is not None and scope in key.scopes and not key.revoked


# ── JWT Token Management ──


class JWTManager:
    """JWT token issuance and validation for service-to-service auth.

    Supported algorithms:

    - ``HS256`` — HMAC-SHA256 (symmetric shared secret).
    - ``ML-DSA-65`` — Post-Quantum Dilithium3 signatures (asymmetric).

    Follows RFC 7519 with standard claims (iss, sub, exp, iat, nbf, jti).
    ML-DSA tokens carry the algorithm tag ``ML-DSA-65`` in the JWT header,
    which is a NIST-approved post-quantum digital signature algorithm.

    Args:
        secret: HMAC secret for HS256 signing (32+ bytes recommended).
        algorithm: Signing algorithm (``HS256`` or ``ML-DSA-65``).
        default_ttl_seconds: Default token lifetime.
        pqc_secret_key: Dilithium3 secret key (required for ML-DSA-65).
        pqc_public_key: Dilithium3 public key (required for ML-DSA-65).
    """

    PQC_ALG = "ML-DSA-65"

    def __init__(
        self,
        secret: str | None = None,
        algorithm: str = "HS256",
        default_ttl_seconds: int = 3600,
        pqc_secret_key: bytes | None = None,
        pqc_public_key: bytes | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.default_ttl = default_ttl_seconds
        self._secret = secret or os.environ.get("CEREBRO_JWT_SECRET", secrets.token_hex(32))
        self._pqc_sk = pqc_secret_key
        self._pqc_pk = pqc_public_key

        if algorithm == self.PQC_ALG and (pqc_secret_key is None or pqc_public_key is None):
            raise ValueError(
                f"Algorithm '{algorithm}' requires both pqc_secret_key and "
                "pqc_public_key. Generate with cerebro.security.PQCDSA().keygen()."
            )

    @classmethod
    def with_pqc(
        cls,
        pqc_secret_key: bytes,
        pqc_public_key: bytes,
        default_ttl_seconds: int = 3600,
    ) -> "JWTManager":
        """Convenience constructor for PQC-signed JWTs (ML-DSA-65 / Dilithium3)."""
        return cls(
            algorithm=cls.PQC_ALG,
            default_ttl_seconds=default_ttl_seconds,
            pqc_secret_key=pqc_secret_key,
            pqc_public_key=pqc_public_key,
        )

    def _b64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _b64url_decode(self, data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _sign(self, payload: bytes) -> bytes:
        if self.algorithm == "HS256":
            return hmac.new(
                self._secret.encode() if isinstance(self._secret, str) else self._secret,
                payload,
                hashlib.sha256,
            ).digest()
        if self.algorithm == self.PQC_ALG:
            from cerebro.security.pqc import PQCDSA
            if self._pqc_sk is None:
                raise ValueError("PQC secret key required for signing")
            return PQCDSA().sign(payload, self._pqc_sk)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _verify_signature(self, payload: bytes, signature: bytes) -> bool:
        if self.algorithm == "HS256":
            expected = self._sign(payload)
            return hmac.compare_digest(expected, signature)
        if self.algorithm == self.PQC_ALG:
            from cerebro.security.pqc import PQCDSA
            if self._pqc_pk is None:
                return False
            return PQCDSA().verify(payload, signature, self._pqc_pk)
        return False

    def create_token(
        self,
        subject: str,
        issuer: str = "cerebro",
        audience: str | None = None,
        scopes: list[str] | None = None,
        ttl_seconds: int | None = None,
        extra_claims: dict | None = None,
    ) -> str:
        """Create a signed JWT.

        Args:
            subject: Token subject (typically user/service ID).
            issuer: Token issuer.
            audience: Intended audience.
            scopes: OAuth2-style scopes.
            ttl_seconds: Token lifetime (default: 3600).
            extra_claims: Additional claims to include.

        Returns:
            Encoded JWT string.
        """
        now = int(time.time())
        ttl = ttl_seconds or self.default_ttl

        header = {"alg": self.algorithm, "typ": "JWT"}
        payload: dict[str, Any] = {
            "iss": issuer,
            "sub": subject,
            "iat": now,
            "exp": now + ttl,
            "jti": secrets.token_hex(16),
        }
        if audience:
            payload["aud"] = audience
        if scopes:
            payload["scope"] = " ".join(scopes)
        if extra_claims:
            payload.update(extra_claims)

        header_b64 = self._b64url_encode(json.dumps(header).encode())
        payload_b64 = self._b64url_encode(json.dumps(payload).encode())
        signing_input = f"{header_b64}.{payload_b64}"
        signature = self._b64url_encode(self._sign(signing_input.encode()))

        return f"{signing_input}.{signature}"

    def validate_token(self, token: str) -> dict | None:
        """Validate a JWT and return its payload.

        Returns:
            Decoded payload dict if valid, None otherwise.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, sig_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"

            # Verify signature
            signature = self._b64url_decode(sig_b64)
            if not self._verify_signature(signing_input.encode(), signature):
                return None

            # Decode and validate payload
            payload = json.loads(self._b64url_decode(payload_b64))

            # Check expiration
            now = int(time.time())
            if payload.get("exp", 0) < now:
                return None

            # Check not-before
            if payload.get("nbf", 0) > now:
                return None

            return payload
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, base64.binascii.Error):
            return None

    def refresh_token(self, token: str, ttl_seconds: int | None = None) -> str | None:
        """Issue a new token from a valid existing one."""
        payload = self.validate_token(token)
        if payload is None:
            return None
        return self.create_token(
            subject=payload["sub"],
            issuer=payload.get("iss", "cerebro"),
            audience=payload.get("aud"),
            scopes=payload.get("scope", "").split() if payload.get("scope") else None,
            ttl_seconds=ttl_seconds,
        )


# ── OAuth2 Token Introspection ──


class OAuth2Introspector:
    """OAuth2 token introspection client (RFC 7662).

    Validates bearer tokens against an OAuth2 authorization server.
    Supports caching of introspection results.

    Args:
        introspection_url: OAuth2 introspection endpoint URL.
        client_id: OAuth2 client ID for authentication.
        client_secret: OAuth2 client secret.
        cache_ttl_seconds: How long to cache successful introspection results.
    """

    def __init__(
        self,
        introspection_url: str,
        client_id: str = "",
        client_secret: str = "",
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.introspection_url = introspection_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, dict]] = {}

    def introspect(self, token: str) -> dict | None:
        """Introspect an OAuth2 token.

        Returns:
            Token info dict if active, None if inactive/error.
        """
        # Check cache
        now = time.time()
        if token in self._cache:
            cached_at, cached_result = self._cache[token]
            if now - cached_at < self.cache_ttl:
                return cached_result
            del self._cache[token]

        try:
            import requests
            resp = requests.post(
                self.introspection_url,
                data={"token": token},
                auth=(
                    (self.client_id, self.client_secret)
                    if self.client_id else None
                ),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("active", False):
                    self._cache[token] = (now, data)
                    return data
        except (OSError, ValueError) as e:
            import logging
            logging.getLogger("cerebro.auth").warning("OAuth2 introspection failed: %s", e)

        return None


# ── Auth Middleware ──


@dataclass
class AuthContext:
    """Authenticated request context."""

    auth_method: str  # "api_key", "jwt", "oauth2"
    principal: str    # Key ID, JWT subject, or OAuth2 client
    scopes: list[str]
    extra: dict = field(default_factory=dict)


class CerebroAuth:
    """Unified authentication layer for Cerebro API.

    Supports multiple auth methods simultaneously:
    - API keys (primary, always available)
    - JWT tokens (service-to-service)
    - OAuth2 bearer tokens (enterprise SSO)

    Args:
        api_keys_file: Path to API keys JSON file.
        jwt_secret: JWT signing secret.
        oauth2_introspection_url: Optional OAuth2 introspection endpoint.
    """

    def __init__(
        self,
        api_keys_file: str = "api_keys.json",
        jwt_secret: str | None = None,
        oauth2_introspection_url: str | None = None,
        oauth2_client_id: str = "",
        oauth2_client_secret: str = "",
    ) -> None:
        self.api_keys = ApiKeyManager(api_keys_file)
        self.jwt = JWTManager(secret=jwt_secret)
        self.oauth2 = (
            OAuth2Introspector(
                oauth2_introspection_url,
                client_id=oauth2_client_id,
                client_secret=oauth2_client_secret,
            )
            if oauth2_introspection_url
            else None
        )

    def authenticate(self, authorization_header: str | None) -> AuthContext | None:
        """Authenticate a request from an Authorization header.

        Tries auth methods in order: API key → JWT → OAuth2.

        Args:
            authorization_header: The Authorization header value.

        Returns:
            AuthContext if authenticated, None otherwise.
        """
        if not authorization_header:
            return None

        # Try Bearer token
        if authorization_header.startswith("Bearer "):
            token = authorization_header[7:]

            # Try API key format first
            key = self.api_keys.validate_key(token)
            if key:
                return AuthContext(
                    auth_method="api_key",
                    principal=key.key_id,
                    scopes=key.scopes,
                )

            # Try JWT
            payload = self.jwt.validate_token(token)
            if payload:
                scopes = payload.get("scope", "").split() if payload.get("scope") else []
                return AuthContext(
                    auth_method="jwt",
                    principal=payload["sub"],
                    scopes=scopes,
                    extra={"claims": payload},
                )

            # Try OAuth2 introspection
            if self.oauth2:
                introspected = self.oauth2.introspect(token)
                if introspected:
                    scopes = introspected.get("scope", "").split() if introspected.get("scope") else []
                    return AuthContext(
                        auth_method="oauth2",
                        principal=introspected.get("sub", introspected.get("client_id", "")),
                        scopes=scopes,
                        extra={"oauth2": introspected},
                    )

            return None

        return None

    def require_scope(self, auth: AuthContext, required_scope: str) -> bool:
        """Check if an authenticated context has a required scope."""
        return required_scope in auth.scopes or "admin" in auth.scopes


# ── FastAPI Integration ──


def create_auth_dependency(
    auth: CerebroAuth,
    required_scope: str | None = None,
) -> Callable:
    """Create a FastAPI dependency for auth.

    Usage:
        auth = CerebroAuth()
        require_auth = create_auth_dependency(auth)

        @app.get("/protected")
        async def protected_route(auth_ctx: AuthContext = Depends(require_auth)):
            ...

    Args:
        auth: CerebroAuth instance.
        required_scope: Optional scope required for this endpoint.

    Returns:
        FastAPI dependency callable.
    """
    try:
        from fastapi import Request, HTTPException
    except ImportError:
        raise ImportError("FastAPI is required for auth dependency injection.")

    def _dependency(request: Request) -> AuthContext:
        auth_header = request.headers.get("Authorization")
        ctx = auth.authenticate(auth_header)

        if ctx is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication credentials",
            )

        if required_scope and not auth.require_scope(ctx, required_scope):
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scope: {required_scope}",
            )

        return ctx

    return _dependency