"""
bqi — Python implementation of the algorithms and quantitative results in
"The Brain as a Query Interface" (Vu, Phenikaa University).

Each submodule corresponds to a section of the paper:

    predictive_coding   -> Sec 3.1  Bayesian Predictive Coding (eq. 1-4, Thm convergence)
    attention            -> Sec 3.3  Attention-based SNIS search (eq. 9-11)
    bqi_algorithm        -> Sec 4.5  Algorithm 1 (BQI three-phase belief update)
    com_algorithm        -> Sec 8.2  Algorithm 2 (Consciousness Optimisation Model)
    iit_phi              -> Sec 5.1  Integrated Information Theory, Phi (eq. 22-23)
    pci                  -> Sec 5.2  Perturbational Complexity Index (eq. 24, Table PCI)
    dmn_ode              -> Sec 7.2  DMN / Phi / QoC ODE system (eq. 25-27, Table ode_params)
    meta_analysis        -> Sec 7.3  Random-effects meta-analysis (16 studies, Table meta)
    hopfield             -> Sec 9.2  Classic / modern Hopfield capacity (Fig retrieval error)
    spectral_graph       -> Sec 4.4 / 9.4 Graph-theoretic brain network + Kuramoto sync
    quantum_decoherence  -> Sec 6    Open quantum system / Orch-OR collapse timescale
"""
__version__ = "1.0.0"
