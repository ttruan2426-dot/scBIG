"""
scBIG Models

- GCAE: Gene-Cluster Aware Encoder/Decoder
- Flow: OT Flow Matching with velocity field
- Networks: Building blocks (MLP, attention, etc.)
"""

from scbig.models.gcae import (
    GeneClusterAwareEncoder,
    GeneClusterAwareDecoder,
    GeneClusterAwareAutoEncoder,
    RelationalPositionalEncoding,
    ModuleInducedAttention,
    GCAETransformerBlock,
    GCAEWrapper,
    GCAE_CONFIGS,
    get_gcae_for_config,
    build_ppi_attention_mask,
    load_ppi_network,
    load_pathway_gene_matrix,
    load_genecompass_embeddings,
    load_genecompass_pe,
)

from scbig.models.flow import (
    ConditionalVelocityField,
    ConditionEncoder,
    OTFlowMatching,
)

from scbig.models.networks import (
    MLPBlock,
    FilmBlock,
    ResNetBlock,
    SelfAttention,
    SelfAttentionBlock,
    SeedAttentionPooling,
    TokenAttentionPooling,
    sinusoidal_time_encoder,
)

__all__ = [
    # GCAE
    "GeneClusterAwareEncoder",
    "GeneClusterAwareDecoder",
    "GeneClusterAwareAutoEncoder",
    "RelationalPositionalEncoding",
    "ModuleInducedAttention",
    "GCAETransformerBlock",
    "GCAEWrapper",
    "GCAE_CONFIGS",
    "get_gcae_for_config",
    "build_ppi_attention_mask",
    "load_ppi_network",
    "load_pathway_gene_matrix",
    "load_genecompass_embeddings",
    "load_genecompass_pe",
    # Flow
    "ConditionalVelocityField",
    "ConditionEncoder",
    "OTFlowMatching",
    # Networks
    "MLPBlock",
    "FilmBlock",
    "ResNetBlock",
    "SelfAttention",
    "SelfAttentionBlock",
    "SeedAttentionPooling",
    "TokenAttentionPooling",
    "sinusoidal_time_encoder",
]
