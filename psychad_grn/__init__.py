"""
psychad_grn: Gene regulatory network analysis of Alzheimer's disease
and cognitive resilience in the PsychAD cohort.

Analyses:
    1. grn_inference       — pySCENIC/GRNboost2 network construction
    2. aucell_scoring      — Regulon activity scoring and concordance
    3. regulon_specificity — RSS computation and robust rank aggregation
    4. dreamlet_analysis   — Differential regulon activity (dreamlet/R)
    5. fishers_enrichment  — Fisher's exact test on DEG × regulon overlap
    6. network_rewiring    — Differential network and GO enrichment
"""

__version__ = "0.1.0"
