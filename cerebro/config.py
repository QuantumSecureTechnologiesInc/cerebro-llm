"""CerebroConfig — all hyperparameters for every Cerebro model tier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CerebroConfig:
    """Complete configuration for a Cerebro model."""

    # ── Tokenizer ──────────────────────────────────────────────
    vocab_size: int = 128_000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Transformer core ──────────────────────────────────────
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4            # GQA ratio 4:1
    head_dim: int | None = None      # auto = hidden_dim // num_heads

    # ── Quaternion path ───────────────────────────────────────
    quaternion_dim: int | None = None  # auto = hidden_dim // 4

    # ── Feed-forward ──────────────────────────────────────────
    ffn_dim: int = 8192             # 4x hidden

    # ── Positional encoding ───────────────────────────────────
    max_seq_len: int = 8192
    rope_theta: float = 10_000.0

    # ── Entropic gating ───────────────────────────────────────
    entropy_min: float = 0.1
    entropy_max: float = 8.0
    init_temperature: float = 1.0

    # ── Bounded recursion (inference only) ────────────────────
    entropy_budget: float = 100.0
    max_recursion_depth: int = 5
    verification_threshold: float = 0.85

    # ── Output head ───────────────────────────────────────────
    logit_soft_cap: float = 50.0    # Gemma-style

    # ── Training defaults ─────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    grad_accum_steps: int = 8
    batch_size: int = 4
    bf16: bool = True

    # ── Reasoning core ────────────────────────────────────────
    reasoning_layers: int = 4

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.quaternion_dim is None:
            self.quaternion_dim = self.hidden_dim // 4

        # ── Validation ──
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed "
                f"num_heads ({self.num_heads})"
            )
        if self.head_dim is not None and self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_heads ({self.num_heads}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        if self.reasoning_layers > self.num_layers:
            raise ValueError(
                f"reasoning_layers ({self.reasoning_layers}) cannot exceed "
                f"num_layers ({self.num_layers})"
            )
        if self.entropy_min >= self.entropy_max:
            raise ValueError(
                f"entropy_min ({self.entropy_min}) must be less than "
                f"entropy_max ({self.entropy_max})"
            )
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {self.ffn_dim}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be at least 1, got {self.max_seq_len}")
        if self.verification_threshold < 0 or self.verification_threshold > 1:
            raise ValueError(
                f"verification_threshold must be between 0 and 1, "
                f"got {self.verification_threshold}"
            )

    # ── Presets ───────────────────────────────────────────────
    @classmethod
    def nano(cls) -> CerebroConfig:
        """Cerebro-Nano: 1.5B params, 24 layers, 2048 hidden, 8K context."""
        return cls(
            vocab_size=128_000,
            hidden_dim=2048,
            num_layers=24,
            num_heads=16,
            num_kv_heads=4,
            ffn_dim=8192,
            max_seq_len=8192,
        )

    @classmethod
    def core(cls) -> CerebroConfig:
        """Cerebro-Core: 7B params, 32 layers, 4096 hidden, 32K context."""
        return cls(
            vocab_size=128_000,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            ffn_dim=16384,
            max_seq_len=32_768,
        )

    @classmethod
    def pro(cls) -> CerebroConfig:
        """Cerebro-Pro: 13B params, 40 layers, 5120 hidden, 64K context."""
        return cls(
            vocab_size=128_000,
            hidden_dim=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            ffn_dim=20480,
            max_seq_len=65_536,
        )

    @classmethod
    def ultra(cls) -> CerebroConfig:
        """Cerebro-Ultra: 34B params, 48 layers, 6144 hidden, 128K context."""
        return cls(
            vocab_size=128_000,
            hidden_dim=6144,
            num_layers=48,
            num_heads=48,
            num_kv_heads=12,
            ffn_dim=24576,
            max_seq_len=131_072,
        )

    @classmethod
    def sovereign(cls) -> CerebroConfig:
        """Cerebro-Sovereign: 70B params, 96 layers, 8192 hidden, 256K context."""
        return cls(
            vocab_size=128_000,
            hidden_dim=8192,
            num_layers=96,
            num_heads=64,
            num_kv_heads=16,
            ffn_dim=32768,
            max_seq_len=262_144,
        )

    @classmethod
    def from_name(cls, name: str) -> CerebroConfig:
        presets = {
            "nano": cls.nano,
            "core": cls.core,
            "pro": cls.pro,
            "ultra": cls.ultra,
            "sovereign": cls.sovereign,
        }
        name = name.lower()
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets)}")
        return presets[name]()
