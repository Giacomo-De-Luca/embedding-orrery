# Psycholinguistic-norm probing — full results (official Glasgow set)

Glasgow: official 5,553-word norms (incl. 871 sense-disambiguated entries),
re-embedded 2026-07-10 (MiniLM ST output; EmbeddingGemma-300m prompt STS;
gemini-embedding-2 SEMANTIC_SIMILARITY). Conc-40k: Brysbaert 39,954 words.
Shared seed-42 80/20 split. MLP = clean 80/20 no-early-stop. Mass-mean R2 is
calibrated (train-split OLS readout); rho is the uncalibrated Spearman.

## Glasgow norms (5,553 words)

### MiniLM-384

| Norm | Ridge R2 | SVR R2 | MLP R2 | MM raw R2 | MM S R2 | MM raw rho | MM S rho | Logit acc |
|---|---|---|---|---|---|---|---|---|
| concreteness | 0.721 | 0.764 | 0.731 | 0.639 | 0.675 | 0.810 | 0.823 | 0.851 |
| imageability | 0.640 | 0.698 | 0.658 | 0.534 | 0.605 | 0.744 | 0.787 | 0.848 |
| valence | 0.617 | 0.695 | 0.661 | 0.490 | 0.528 | 0.688 | 0.733 | 0.761 |
| aoa | 0.576 | 0.631 | 0.572 | 0.440 | 0.534 | 0.655 | 0.732 | 0.772 |
| semsize | 0.601 | 0.675 | 0.631 | 0.488 | 0.565 | 0.700 | 0.759 | 0.782 |
| gender | 0.489 | 0.586 | 0.534 | 0.366 | 0.423 | 0.598 | 0.651 | 0.752 |
| familiarity | 0.455 | 0.522 | 0.421 | 0.356 | 0.419 | 0.622 | 0.683 | 0.761 |
| arousal | 0.483 | 0.567 | 0.496 | 0.358 | 0.429 | 0.580 | 0.655 | 0.730 |
| dominance | 0.463 | 0.547 | 0.483 | 0.392 | 0.390 | 0.603 | 0.613 | 0.710 |

### EmbGemma-768

| Norm | Ridge R2 | SVR R2 | MLP R2 | MM raw R2 | MM S R2 | MM raw rho | MM S rho | Logit acc |
|---|---|---|---|---|---|---|---|---|
| concreteness | 0.841 | 0.860 | 0.824 | 0.736 | 0.794 | 0.864 | 0.885 | 0.879 |
| imageability | 0.775 | 0.800 | 0.777 | 0.672 | 0.737 | 0.827 | 0.859 | 0.869 |
| valence | 0.817 | 0.857 | 0.848 | 0.712 | 0.654 | 0.817 | 0.824 | 0.817 |
| aoa | 0.681 | 0.706 | 0.679 | 0.347 | 0.630 | 0.581 | 0.802 | 0.815 |
| semsize | 0.742 | 0.786 | 0.756 | 0.529 | 0.667 | 0.723 | 0.829 | 0.819 |
| gender | 0.711 | 0.753 | 0.726 | 0.418 | 0.565 | 0.690 | 0.760 | 0.780 |
| familiarity | 0.596 | 0.589 | 0.584 | 0.316 | 0.545 | 0.599 | 0.773 | 0.788 |
| arousal | 0.656 | 0.707 | 0.684 | 0.504 | 0.582 | 0.697 | 0.768 | 0.782 |
| dominance | 0.649 | 0.689 | 0.663 | 0.456 | 0.518 | 0.658 | 0.718 | 0.750 |

### Gemini-3072

| Norm | Ridge R2 | SVR R2 | MLP R2 | MM raw R2 | MM S R2 | MM raw rho | MM S rho | Logit acc |
|---|---|---|---|---|---|---|---|---|
| concreteness | 0.811 | 0.853 | 0.846 | 0.733 | 0.756 | 0.868 | 0.867 | 0.873 |
| imageability | 0.755 | 0.805 | 0.795 | 0.663 | 0.713 | 0.827 | 0.848 | 0.853 |
| valence | 0.750 | 0.831 | 0.798 | 0.632 | 0.594 | 0.773 | 0.791 | 0.790 |
| aoa | 0.710 | 0.786 | 0.753 | 0.491 | 0.624 | 0.706 | 0.806 | 0.797 |
| semsize | 0.687 | 0.775 | 0.724 | 0.463 | 0.589 | 0.686 | 0.793 | 0.814 |
| gender | 0.596 | 0.732 | 0.682 | 0.457 | 0.495 | 0.702 | 0.734 | 0.772 |
| familiarity | 0.577 | 0.673 | 0.646 | 0.499 | 0.551 | 0.731 | 0.766 | 0.761 |
| arousal | 0.550 | 0.707 | 0.642 | 0.457 | 0.508 | 0.654 | 0.724 | 0.757 |
| dominance | 0.539 | 0.669 | 0.637 | 0.445 | 0.455 | 0.641 | 0.690 | 0.743 |

## Concreteness-40k (39,954 words)

| Model | Ridge R2 | SVR R2 | MLP R2 | MM raw R2 | MM S R2 | MM raw rho | MM S rho | Logit acc |
|---|---|---|---|---|---|---|---|---|
| MiniLM-384 | 0.700 | 0.744 | 0.724 | 0.615 | 0.678 | 0.807 | 0.834 | 0.846 |
| EmbGemma-768 | 0.832 | 0.842 | 0.846 | 0.707 | 0.790 | 0.858 | 0.894 | 0.890 |
| Gemini-3072 | 0.845 | 0.848 | 0.846 | 0.640 | 0.786 | 0.816 | 0.892 | 0.884 |
