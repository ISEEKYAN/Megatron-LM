# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact typed coverage binding for the closed precision profiles."""

from __future__ import annotations

from dataclasses import dataclass

from megatron.lite.primitive.precision.contract import (
    PrecisionImplementation,
    PrimitiveCapability,
    SemanticSite,
)
from megatron.lite.primitive.precision.hopper_blockwise import (
    PrecisionPhase,
    active_precision,
)

# Each FP8 capability may only be claimed by the exact Transformer Engine GEMM
# primitive that implements it. Resolving the class lazily keeps this module
# importable on hosts without Transformer Engine (CPU unit tests).
_CAPABILITY_TE_ATTR: dict[PrimitiveCapability, str] = {
    PrimitiveCapability.TE_LINEAR: "Linear",
    PrimitiveCapability.TE_LAYERNORM_LINEAR: "LayerNormLinear",
    PrimitiveCapability.TE_GROUPED_LINEAR: "GroupedLinear",
}


def _te_primitive_type(capability: PrimitiveCapability) -> type:
    """Return the TE primitive class that a capability must be backed by."""

    import transformer_engine.pytorch as te  # local import: keep CPU import-safety

    return getattr(te, _CAPABILITY_TE_ATTR[capability])


@dataclass(frozen=True, slots=True)
class _Requirement:
    owner: object
    site: SemanticSite
    capabilities: frozenset[PrimitiveCapability]
    diagnostic: str


@dataclass(frozen=True, slots=True)
class _Claim:
    owner: object
    site: SemanticSite
    capability: PrimitiveCapability
    diagnostic: str


@dataclass(frozen=True, slots=True)
class CoverageEntry:
    """One sealed semantic site and the primitive capability bound to it."""

    object_id: int
    site: SemanticSite
    capability: PrimitiveCapability | None
    diagnostic: str


@dataclass(frozen=True, slots=True)
class CoverageManifest:
    """Immutable proof that every declared concrete site was bound exactly once."""

    implementation_name: str
    entries: tuple[CoverageEntry, ...]


class PrecisionCoverage:
    """Collect typed requirements and claims during one model construction."""

    __slots__ = ("_claims", "_implementation", "_manifest", "_requirements")

    def __init__(self, implementation: PrecisionImplementation):
        if not isinstance(implementation, PrecisionImplementation):
            raise TypeError("coverage requires a PrecisionImplementation")
        self._implementation = implementation
        self._requirements: list[_Requirement] = []
        self._claims: list[_Claim] = []
        self._manifest: CoverageManifest | None = None

    def _ensure_collecting(self) -> None:
        if self._manifest is not None:
            raise RuntimeError("precision coverage is already sealed")
        if active_precision(PrecisionPhase.MODEL_INIT) is not self._implementation:
            raise RuntimeError(
                "precision coverage is bound to a different model-init context"
            )

    @staticmethod
    def _validate_owner_site(owner: object, site: SemanticSite) -> None:
        if owner is None:
            raise TypeError("coverage owner cannot be None")
        if not isinstance(site, SemanticSite):
            raise TypeError("coverage site must be a SemanticSite")

    def require(
        self,
        owner: object,
        site: SemanticSite,
        capabilities: frozenset[PrimitiveCapability] = frozenset(),
        *,
        diagnostic: str = "",
    ) -> None:
        """Declare one concrete FP8 site or fixed-BF16 exclusion."""

        self._ensure_collecting()
        self._validate_owner_site(owner, site)
        if not isinstance(capabilities, frozenset) or not all(
            isinstance(capability, PrimitiveCapability) for capability in capabilities
        ):
            raise TypeError(
                "coverage capabilities must be a frozenset of PrimitiveCapability"
            )
        if site in self._implementation.fp8_sites and not capabilities:
            raise ValueError(
                f"FP8 site {site.value} must declare compatible primitive capabilities"
            )
        if site in self._implementation.bf16_sites and capabilities:
            raise ValueError(
                f"fixed BF16 site {site.value} cannot accept an FP8 capability"
            )
        self._requirements.append(_Requirement(owner, site, capabilities, diagnostic))

    def claim(
        self,
        owner: object,
        site: SemanticSite,
        capability: PrimitiveCapability,
        primitive: object = None,
        *,
        diagnostic: str = "",
    ) -> None:
        """Claim one exact concrete site, backed by a real TE primitive.

        ``owner`` is the identity matched against the declared requirement.
        ``primitive`` is the object whose type must actually implement the
        capability; it defaults to ``owner`` for primitives that expose the TE
        module directly (dense MLP, MoE experts) and is passed explicitly by
        wrappers whose owner is not itself the TE module (TP linears). The
        type check makes a capability unforgeable: a non-TE module (e.g. a raw
        ``torch.nn.Linear`` or an SDPA/matmul owner) cannot masquerade as an
        FP8-capable GEMM site.
        """

        self._ensure_collecting()
        self._validate_owner_site(owner, site)
        if not isinstance(capability, PrimitiveCapability):
            raise TypeError("coverage capability must be a PrimitiveCapability")
        witness = owner if primitive is None else primitive
        expected = _te_primitive_type(capability)
        if not isinstance(witness, expected):
            actual = type(witness)
            raise TypeError(
                f"capability {capability.value} at {site.value} must be backed by a "
                f"real transformer_engine.pytorch.{_CAPABILITY_TE_ATTR[capability]} "
                f"primitive; got {actual.__module__}.{actual.__qualname__}"
            )
        self._claims.append(_Claim(owner, site, capability, diagnostic))

    @property
    def implementation(self) -> PrecisionImplementation:
        """Return the closed implementation this collector is bound to."""

        return self._implementation

    @property
    def manifest(self) -> CoverageManifest:
        if self._manifest is None:
            raise RuntimeError(
                "precision coverage was not sealed before optimizer construction"
            )
        return self._manifest

    @staticmethod
    def _key(item: _Requirement | _Claim) -> tuple[int, SemanticSite]:
        return (id(item.owner), item.site)

    def seal(self) -> CoverageManifest:
        """Match requirements and claims exactly, then make the manifest immutable."""

        self._ensure_collecting()
        requirement_groups: dict[tuple[int, SemanticSite], list[_Requirement]] = {}
        claim_groups: dict[tuple[int, SemanticSite], list[_Claim]] = {}
        for requirement in self._requirements:
            requirement_groups.setdefault(self._key(requirement), []).append(
                requirement
            )
        for claim in self._claims:
            claim_groups.setdefault(self._key(claim), []).append(claim)

        errors: list[str] = []
        for requirements in requirement_groups.values():
            if len(requirements) > 1:
                errors.append(
                    f"duplicate requirement for {requirements[0].site.value} "
                    f"({requirements[0].diagnostic})"
                )
        for claims in claim_groups.values():
            if len(claims) > 1:
                errors.append(
                    f"duplicate claim for {claims[0].site.value} ({claims[0].diagnostic})"
                )

        entries: list[CoverageEntry] = []
        for requirement in self._requirements:
            key = self._key(requirement)
            claims = claim_groups.get(key, [])
            if requirement.site in self._implementation.bf16_sites:
                if claims:
                    errors.append(
                        f"fixed BF16 site {requirement.site.value} received an FP8 claim "
                        f"({requirement.diagnostic})"
                    )
                entries.append(
                    CoverageEntry(
                        id(requirement.owner),
                        requirement.site,
                        None,
                        requirement.diagnostic,
                    )
                )
                continue
            if not claims:
                errors.append(
                    f"missing claim for {requirement.site.value} ({requirement.diagnostic})"
                )
                continue
            claim = claims[0]
            if claim.capability not in requirement.capabilities:
                errors.append(
                    f"incompatible capability {claim.capability.value} for {requirement.site.value} "
                    f"({requirement.diagnostic})"
                )
                continue
            entries.append(
                CoverageEntry(
                    id(requirement.owner),
                    requirement.site,
                    claim.capability,
                    requirement.diagnostic or claim.diagnostic,
                )
            )

        for key, claims in claim_groups.items():
            if key not in requirement_groups:
                claim = claims[0]
                errors.append(
                    f"unconsumed claim for {claim.site.value} ({claim.diagnostic})"
                )

        if errors:
            raise ValueError("precision coverage failed: " + "; ".join(errors))
        self._manifest = CoverageManifest(self._implementation.name, tuple(entries))
        return self._manifest
