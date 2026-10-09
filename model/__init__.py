"""
model/ - Q1_pipeline_model.md implementation package.

Five result packages:
  theory          Goi 1: Propositions P1-P5 (M0 model errors + M1 stationary properties)
  identifiability Goi 2: Structural and practical identifiability, Fisher/CRB
  design          Goi 3: Experimental design map (CRB grid + MLE representative cells)
  control         Goi 4: LQG vs bang-bang closed-loop control + robustness
  sensitivity     Goi 5: Sobol/Saltelli global sensitivity analysis

All data is generated in silico from StateSpaceDMN (bqi.dmn_ode) with fixed seeds.
Reproduce all paper figures/tables: python main.py model all --seed 42
"""
