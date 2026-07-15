"""CerebroConfig — all hyperparameters for every Cerebro model tier.

Cerebro ships six presets that form a **feature ladder**: each tier adds
capabilities on top of the smaller one, so the flagship ``sovereign``
preset unlocks the most advanced feature set (MoE, deep reasoning core,
long-context RoPE scaling, multimodal, watermarking, mandatory PQC, etc.).

+──────────────────+──────+──────+──────+──────+───────+───────────+
| Feature          | Tiny | Nano | Core | Pro  | Ultra | Sovereign |
+──────────────────+──────+──────+──────+──────+───────+───────────+
| Quaternion attn  |  ✗   |  ✓   |  ✓   |  ✓   |  ✓    |    ✓     |
| Entropic gating  |  ✓   |  ✓   |  ✓   |  ✓   |  ✓    |    ✓     |
| Reasoning layers |  2   |  4   |  6   |  8   | 10    |   12     |
| Recursion depth  |  2   |  3   |  5   |  6   |  8    |   12     |
| Self-verify      |  ✗   |  ✗   |  ✓   |  ✓   |  ✓    |    ✓     |
| MoE FFN          |  ✗   |  ✗   |  ✗   | 8×k2 | 16×k2 |  32×k2   |
| RoPE θ           | 10K  | 10K  | 500K |  1M  |  2M   |   8M     |
| KV-cache bits    |  16  |  16  |   8  |   8  |   4   |    4     |
| Speculative dec  |  ✗   |  ✗   |  ✗   |  ✓   |  ✓    |    ✓     |
| Vision multi-mod |  ✗   |  ✗   |  ✗   |  ✓   |  ✓    |    ✓     |
| Constitutional AI|  ✗   |  ✗   |  ✓   |  ✓   |  ✓    |    ✓     |
| Tool use / RAG   |  ✗   |  ✗   |  ✓   |  ✓   |  ✓    |    ✓     |
| Output watermark |  ✗   |  ✗   |  ✗   |  ✗   |  ✓    |    ✓     |
| PQC required     |  ✗   |  ✗   |  opt |  opt |  ✓    |    ✓     |
+──────────────────+──────+──────+──────+──────+───────+───────────+
"""

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
    use_quaternion_attention: bool = True  # Tiny disables for CPU speed

    # ── Feed-forward ──────────────────────────────────────────
    ffn_dim: int = 8192              # 4x hidden

    # ── Mixture-of-Experts (Pro/Ultra/Sovereign) ──────────────
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25
    moe_aux_loss_scale: float = 1e-2   # weight of load-balance aux loss

    # ── Positional encoding ───────────────────────────────────
    max_seq_len: int = 8192
    rope_theta: float = 10_000.0
    long_context_scaling: float = 1.0   # NTK/YaRN scaling factor for Ultra+

    # ── Entropic gating ───────────────────────────────────────
    entropy_min: float = 0.1
    entropy_max: float = 8.0
    init_temperature: float = 1.0

    # ── Reasoning core ────────────────────────────────────────
    reasoning_layers: int = 4
    use_self_verification: bool = False   # Core+ enables verification head

    # ── Bounded recursion (inference only) ────────────────────
    entropy_budget: float = 100.0
    max_recursion_depth: int = 5
    verification_threshold: float = 0.85

    # ── KV-cache ──────────────────────────────────────────────
    kv_cache_bits: int = 16              # 16=fp16/bf16, 8=int8, 4=int4

    # ── Inference features ────────────────────────────────────
    enable_speculative_decoding: bool = False
    enable_vision: bool = False
    enable_tool_use: bool = False
    enable_rag: bool = False
    enable_watermarking: bool = False

    # ── Alignment / training features ─────────────────────────
    enable_constitutional_ai: bool = False

    # ── Output head ───────────────────────────────────────────
    logit_soft_cap: float = 50.0     # Gemma-style

    # ── Training defaults ─────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    grad_accum_steps: int = 8
    batch_size: int = 4
    bf16: bool = True

    # ── Security ──────────────────────────────────────────────
    require_pqc: bool = False        # Ultra+ refuses to load unsigned weights

    # ── Tier label ────────────────────────────────────────────
    tier: str = "custom"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.quaternion_dim is None:
            self.quaternion_dim = self.hidden_dim // 4

        # ── Structural validation ──
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

        # ── Feature validation ──
        if self.kv_cache_bits not in (4, 8, 16):
            raise ValueError(
                f"kv_cache_bits must be one of (4, 8, 16), got {self.kv_cache_bits}"
            )
        if self.use_moe:
            if self.moe_num_experts < 2:
                raise ValueError(
                    f"moe_num_experts must be >= 2 when use_moe=True, "
                    f"got {self.moe_num_experts}"
                )
            if self.moe_top_k < 1 or self.moe_top_k > self.moe_num_experts:
                raise ValueError(
                    f"moe_top_k ({self.moe_top_k}) must be in "
                    f"[1, {self.moe_num_experts}]"
                )
        if self.long_context_scaling <= 0:
            raise ValueError(
                f"long_context_scaling must be positive, got {self.long_context_scaling}"
            )

        # ── Quaternion algebra requires 4-way divisibility ──
        # Quaternion projections (QLinear) reshape hidden_dim into groups
        # of 4 real components; if the divisibility invariants fail the
        # model will construct successfully but produce garbage at
        # runtime. Validate up front.
        if self.use_quaternion_attention:
            if self.hidden_dim % 4 != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by 4 "
                    f"when use_quaternion_attention=True"
                )
            if self.head_dim is not None and self.head_dim % 4 != 0:
                raise ValueError(
                    f"head_dim ({self.head_dim}) must be divisible by 4 "
                    f"when use_quaternion_attention=True"
                )

    # ── Feature summary (introspection) ───────────────────────
    def enabled_features(self) -> dict[str, bool | int | str]:
        """Return a dict of enabled features for logging/UI display."""
        return {
            "tier": self.tier,
            "quaternion_attention": self.use_quaternion_attention,
            "moe": self.use_moe,
            "moe_experts": self.moe_num_experts if self.use_moe else 0,
            "moe_top_k": self.moe_top_k if self.use_moe else 0,
            "reasoning_layers": self.reasoning_layers,
            "self_verification": self.use_self_verification,
            "max_recursion_depth": self.max_recursion_depth,
            "kv_cache_bits": self.kv_cache_bits,
            "speculative_decoding": self.enable_speculative_decoding,
            "vision": self.enable_vision,
            "tool_use": self.enable_tool_use,
            "rag": self.enable_rag,
            "watermarking": self.enable_watermarking,
            "constitutional_ai": self.enable_constitutional_ai,
            "require_pqc": self.require_pqc,
            "context_length": self.max_seq_len,
            "rope_theta": self.rope_theta,
        }

    # ── Presets ───────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> CerebroConfig:
        """Cerebro-Tiny: ~350M params, 12 layers, 768 hidden, 2K context.

        Baseline for CPU testing and unit tests. Disables quaternion
        attention for maximum throughput on non-GPU hardware.
        """
        return cls(
            tier="tiny",
            vocab_size=128_000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            ffn_dim=3072,
            max_seq_len=2048,
            batch_size=2,
            learning_rate=5e-4,
            warmup_steps=500,
            max_steps=10_000,
            # Feature ladder
            use_quaternion_attention=False,
            reasoning_layers=2,
            max_recursion_depth=2,
            use_self_verification=False,
        )

    @classmethod
    def nano(cls) -> CerebroConfig:
        """Cerebro-Nano: 1.5B params, 24 layers, 2048 hidden, 8K context.

        Edge deployment tier. Enables quaternion attention for the first
        time, keeps other capabilities minimal for latency.
        """
        return cls(
            tier="nano",
            vocab_size=128_000,
            hidden_dim=2048,
            num_layers=24,
            num_heads=16,
            num_kv_heads=4,
            ffn_dim=8192,
            max_seq_len=8192,
            # Feature ladder
            use_quaternion_attention=True,
            reasoning_layers=4,
            max_recursion_depth=3,
            use_self_verification=False,
        )

    @classmethod
    def core(cls) -> CerebroConfig:
        """Cerebro-Core: 7B params, 32 layers, 4096 hidden, 32K context.

        General-purpose tier. Turns on self-verification, constitutional
        AI, tool use, RAG, and int8 KV-cache. Long-context RoPE scaling
        begins here.
        """
        return cls(
            tier="core",
            vocab_size=128_000,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            ffn_dim=16384,
            max_seq_len=32_768,
            rope_theta=500_000.0,
            # Feature ladder
            use_quaternion_attention=True,
            reasoning_layers=6,
            max_recursion_depth=5,
            use_self_verification=True,
            kv_cache_bits=8,
            enable_constitutional_ai=True,
            enable_tool_use=True,
            enable_rag=True,
        )

    @classmethod
    def pro(cls) -> CerebroConfig:
        """Cerebro-Pro: 13B params, 40 layers, 5120 hidden, 64K context.

        Professional tier. Introduces **MoE** (8 experts × top-2),
        speculative decoding, and multimodal vision.
        """
        return cls(
            tier="pro",
            vocab_size=128_000,
            hidden_dim=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            ffn_dim=20480,
            max_seq_len=65_536,
            rope_theta=1_000_000.0,
            # Feature ladder
            use_quaternion_attention=True,
            reasoning_layers=8,
            max_recursion_depth=6,
            use_self_verification=True,
            kv_cache_bits=8,
            use_moe=True,
            moe_num_experts=8,
            moe_top_k=2,
            enable_constitutional_ai=True,
            enable_tool_use=True,
            enable_rag=True,
            enable_speculative_decoding=True,
            enable_vision=True,
        )

    @classmethod
    def ultra(cls) -> CerebroConfig:
        """Cerebro-Ultra: 34B params, 48 layers, 6144 hidden, 128K context.

        Research tier. Scales MoE to 16 experts, activates int4 KV-cache,
        output watermarking, and NTK-scaled long-context RoPE. Requires
        PQC-signed weights at load time.
        """
        return cls(
            tier="ultra",
            vocab_size=128_000,
            hidden_dim=6144,
            num_layers=48,
            num_heads=48,
            num_kv_heads=12,
            ffn_dim=24576,
            max_seq_len=131_072,
            rope_theta=2_000_000.0,
            long_context_scaling=2.0,
            # Feature ladder
            use_quaternion_attention=True,
            reasoning_layers=10,
            max_recursion_depth=8,
            use_self_verification=True,
            kv_cache_bits=4,
            use_moe=True,
            moe_num_experts=16,
            moe_top_k=2,
            enable_constitutional_ai=True,
            enable_tool_use=True,
            enable_rag=True,
            enable_speculative_decoding=True,
            enable_vision=True,
            enable_watermarking=True,
            require_pqc=True,
        )

    @classmethod
    def sovereign(cls) -> CerebroConfig:
        """Cerebro-Sovereign: 70B params, 96 layers, 8192 hidden, 256K context.

        Enterprise flagship. **Every** advanced feature enabled: 32-expert
        MoE, maximum reasoning depth, deepest bounded recursion, int4
        KV-cache, watermarking, mandatory PQC-signed weights, and full
        YaRN-scaled long-context RoPE.
        """
        return cls(
            tier="sovereign",
            vocab_size=128_000,
            hidden_dim=8192,
            num_layers=96,
            num_heads=64,
            num_kv_heads=16,
            ffn_dim=32768,
            max_seq_len=262_144,
            rope_theta=8_000_000.0,
            long_context_scaling=4.0,
            # Feature ladder — all-on
            use_quaternion_attention=True,
            reasoning_layers=12,
            max_recursion_depth=12,
            use_self_verification=True,
            kv_cache_bits=4,
            use_moe=True,
            moe_num_experts=32,
            moe_top_k=2,
            enable_constitutional_ai=True,
            enable_tool_use=True,
            enable_rag=True,
            enable_speculative_decoding=True,
            enable_vision=True,
            enable_watermarking=True,
            require_pqc=True,
            # Tighter alignment for flagship
            entropy_budget=200.0,
            verification_threshold=0.90,
        )

    @classmethod
    def from_name(cls, name: str) -> CerebroConfig:
        presets = {
            "tiny": cls.tiny,
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
