# Intelligent Test Pattern Optimization for Digital Circuit Testing

## What's included
- `src/main.py`: complete circuit simulation, stuck-at fault injection, Random Forest ranking, greedy coverage selection, and baseline comparisons.
- `data/`: generated test patterns, target faults, and fault-detection matrix.
- `results/`: AI ranking, comparison table, coverage curve, random-baseline runs, and machine-readable summary.
- `docs/`: project report and presentation.
- `requirements.txt`: Python dependencies.

## Run it
Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python src/main.py
```

Optional arguments:
```bash
python src/main.py --seed 7 --target-coverage 0.90
```

## Circuit and fault model
Inputs are `a,b,c,d`; outputs are `y0,y1,y2`.
- `n1 = a XOR b`
- `n2 = c AND d`
- `n3 = a OR c`
- `n4 = b NAND d`
- `y0 = n1 XOR n2`
- `y1 = n3 AND n4`
- `y2 = y0 OR y1`

Target faults are single stuck-at-0 and stuck-at-1 faults on `n1,n2,n3,n4,y0,y1,y2` (14 target faults). A pattern detects a fault when at least one observable output differs from the fault-free circuit output.

## AI method
A RandomForestRegressor learns a pattern-level fault-detection score using input-bit features, number of ones, bit transitions, and balanced-bit indicator. Patterns are ranked by predicted score. A greedy selector adds patterns that detect previously undetected faults until the target coverage is reached or the candidate list is exhausted.

The model is an educational ranking aid, not a production ATPG engine. The exhaustive 16-pattern dataset is deliberately small; cross-validation is included as a diagnostic only.

## Baselines
- Binary-order greedy selection.
- 100 seeded random-order greedy selections, summarized in `results/strategy_comparison.csv`.

## Important interpretation
Compare actual fault coverage and pattern count, not only the AI score. On a tiny circuit, AI may not outperform the baseline; report the observed results honestly. Runtime is small and can be measured with the Python shell or OS timer. For a larger research project, add benchmark circuits and richer structural features.

## Submission note
The report and presentation are generated from the included experiment. Add your personal/college details and review the results before submitting. The AI approach is a demonstration and is not guaranteed to beat the baselines on this small circuit.
