# Approaches (plain English log)

- [v0](v0/README.md): mass filter + spectral cosine. LB 0.088. Best so far.
- [v1](v1/README.md): subformula explained-intensity. LB 0.044. Confidently wrong on novel molecules.
- [v2](v2/README.md): fingerprints (memory blend + learned MLP). Local MRR 0.047 / 0.038. Ranker isn't the cap.
- [v3](v3/README.md): (in progress) train + COCONUT candidates under v0-cosine. Targets the recall cap.
- [Kernel](kernels/baseline-v0/README.md): how local code becomes the Kaggle notebook.
- [Constraints + papers](constraints-and-literature.md): what binds winning and why.
