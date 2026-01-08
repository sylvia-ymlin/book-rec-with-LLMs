# Temporal Dynamics Benchmark Report
**Date**: 2026-01-08
**Mechanism**: Recency Boosting (Log-Linear Decay).

## Experiment Setup
- **Query**: "latest advancements in technology and science"
- **Method**: Compare Rerank Score vs Temporal Boosted Score.
- **Boost Logic**: `Score_New = Score_Old + (2.0 / log(Age + e))`

## Results Comparison

| Title (Year) | Standard Score | Temporal Score | Boost | Age |
| :--- | :--- | :--- | :--- | :--- |
| **"Intro to Science..." (2011)** | 6.076 | **6.772** | +0.696 | 15 yrs |
| **"Environmental Sci..." (2012)** | -0.883 | **-0.173** | +0.710 | 14 yrs |
| **"ACP Complete..." (1999)** | 4.128 | 4.718 | +0.590 | 27 yrs |

## Analysis
- **Correlation**: Newer books receive a higher additive boost.
- **Magnitude**: ~0.7 points for a 15-year-old book vs ~0.59 for a 27-year-old book.
- **Impact**: Enough to tip the scales in close calls or move a "relevant but old" book below a "relevant and new" one.
- **Safety**: Does NOT bury classic books (1999 still retained high rank due to high base relevance).

## Conclusion
Temporal Dynamics successfully implements a "Freshness Bias" without compromising semantic relevance.
