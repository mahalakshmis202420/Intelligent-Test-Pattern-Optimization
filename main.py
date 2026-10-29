"""
Capstone 10: Intelligent Test Pattern Optimization for Digital Circuit Testing
Educational reference implementation.

Circuit: 4-input combinational logic network with 3 outputs.
Fault model: single stuck-at-0 / stuck-at-1 faults on selected internal and output nets.
Pattern space: all 16 input combinations.
AI: RandomForestRegressor ranks patterns by fault-detection score; a greedy coverage
selector uses predicted scores as a tie-breaker while maximizing newly detected faults.
Baseline: seeded random ordering, then greedy prefix evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Gate-level network. Every node is a Boolean signal.
# Inputs: a,b,c,d
# n1 = a XOR b; n2 = c AND d; n3 = a OR c; n4 = b NAND d
# y0 = n1 XOR n2; y1 = n3 AND n4; y2 = y0 OR y1
INPUTS = ["a", "b", "c", "d"]
INTERNAL_NODES = ["n1", "n2", "n3", "n4", "y0", "y1", "y2"]
FAULT_NODES = INTERNAL_NODES

def evaluate(bits, fault=None):
    """Evaluate circuit. fault=(node, stuck_value) optionally injects a stuck-at fault."""
    v = dict(zip(INPUTS, [int(x) for x in bits]))
    def assign(name, value):
        if fault and fault[0] == name:
            v[name] = int(fault[1])
        else:
            v[name] = int(value)
    assign("n1", v["a"] ^ v["b"])
    assign("n2", v["c"] & v["d"])
    assign("n3", v["a"] | v["c"])
    assign("n4", 1 - (v["b"] & v["d"]))  # NAND
    assign("y0", v["n1"] ^ v["n2"])
    assign("y1", v["n3"] & v["n4"])
    assign("y2", v["y0"] | v["y1"])
    return tuple(v[y] for y in ["y0", "y1", "y2"])

def all_patterns():
    return [tuple((i >> shift) & 1 for shift in (3,2,1,0)) for i in range(16)]

def all_faults():
    return [(node, stuck) for node in FAULT_NODES for stuck in (0,1)]

def build_fault_matrix():
    patterns = all_patterns()
    faults = all_faults()
    rows = []
    for p in patterns:
        good = evaluate(p)
        row = []
        for f in faults:
            row.append(int(evaluate(p, f) != good))
        rows.append(row)
    return patterns, faults, np.asarray(rows, dtype=int)

def feature_frame(patterns, matrix):
    arr = np.asarray(patterns, dtype=int)
    df = pd.DataFrame(arr, columns=[f"x_{n}" for n in INPUTS])
    df["ones_count"] = arr.sum(axis=1)
    df["transitions"] = np.abs(np.diff(arr, axis=1)).sum(axis=1)
    df["balanced_bits"] = ((arr.sum(axis=1) >= 1) & (arr.sum(axis=1) <= 3)).astype(int)
    df["faults_detected"] = matrix.sum(axis=1)
    df["coverage_fraction"] = df["faults_detected"] / matrix.shape[1]
    return df

def greedy_select(order, matrix, target=0.90):
    """Select from supplied pattern order; maximize new fault coverage, order breaks ties."""
    uncovered = set(range(matrix.shape[1]))
    chosen = []
    for idx in order:
        newly = [j for j in uncovered if matrix[idx, j] == 1]
        if newly:
            chosen.append(int(idx))
            uncovered.difference_update(newly)
        if 1 - len(uncovered)/matrix.shape[1] >= target:
            break
    return chosen

def evaluate_selection(selected, matrix):
    if not selected:
        return 0.0, 0, matrix.shape[1]
    detected = np.any(matrix[selected, :] == 1, axis=0)
    return float(detected.mean()), int(detected.sum()), int(matrix.shape[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

    patterns, faults, matrix = build_fault_matrix()
    fault_names = [f"{n}_SA{v}" for n,v in faults]
    pattern_names = ["".join(map(str,p)) for p in patterns]
    fm = pd.DataFrame(matrix, index=pattern_names, columns=fault_names)
    fm.index.name = "pattern"
    fm.to_csv(DATA / "fault_detection_matrix.csv")

    # Save complete pattern and fault descriptions.
    pd.DataFrame([{"pattern": "".join(map(str,p)), **{f"x_{n}":p[i] for i,n in enumerate(INPUTS)},
                   "good_outputs": "".join(map(str,evaluate(p)))} for p in patterns]).to_csv(
        DATA / "generated_test_patterns.csv", index=False)
    pd.DataFrame([{"fault": f"{n}_SA{v}", "node":n, "stuck_at":v} for n,v in faults]).to_csv(
        DATA / "target_faults.csv", index=False)

    features = feature_frame(patterns, matrix)
    X = features[["x_a","x_b","x_c","x_d","ones_count","transitions","balanced_bits"]]
    y = features["faults_detected"]
    model = RandomForestRegressor(n_estimators=250, random_state=args.seed, max_depth=4,
                                  min_samples_leaf=1)
    # Small exhaustive dataset: train on all 16 vectors to produce a practical ordering.
    # CV score is reported as a rough educational diagnostic, not a generalization claim.
    cv = KFold(n_splits=4, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    model.fit(X, y)
    features["rf_predicted_score"] = model.predict(X)
    features["actual_detected_faults"] = y
    features["rf_rank"] = features["rf_predicted_score"].rank(method="first", ascending=False).astype(int)
    features["pattern"] = pattern_names
    features.to_csv(RESULTS / "ai_pattern_ranking.csv", index=False)

    # AI-ranked greedy selection: model rank guides the order; greedy coverage removes redundancy.
    ai_order = np.argsort(-features["rf_predicted_score"].to_numpy())
    ai_selected = greedy_select(ai_order, matrix, args.target_coverage)

    # Basic deterministic baseline: test patterns in binary enumeration order.
    basic_selected = greedy_select(list(range(len(patterns))), matrix, args.target_coverage)

    # Random baseline averaged across 100 independent seeded permutations.
    rng = np.random.default_rng(args.seed)
    random_rows = []
    for run in range(100):
        order = rng.permutation(len(patterns))
        selected = greedy_select(order, matrix, args.target_coverage)
        cov, count, total = evaluate_selection(selected, matrix)
        random_rows.append({"run":run+1, "patterns_used":len(selected), "fault_coverage":cov,
                            "faults_detected":count, "total_faults":total})
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(RESULTS / "random_baseline_runs.csv", index=False)

    rows = []
    for name, selected in [("AI_RandomForest_ranked_greedy", ai_selected),
                           ("Basic_binary_order_greedy", basic_selected)]:
        cov, count, total = evaluate_selection(selected, matrix)
        rows.append({"strategy":name, "patterns_used":len(selected), "fault_coverage":cov,
                     "faults_detected":count, "total_faults":total,
                     "selected_patterns":";".join(pattern_names[i] for i in selected)})
    rows.append({"strategy":"Random_order_greedy_mean_100_runs",
                 "patterns_used":round(random_df.patterns_used.mean(),2),
                 "fault_coverage":random_df.fault_coverage.mean(),
                 "faults_detected":round(random_df.faults_detected.mean(),2),
                 "total_faults":matrix.shape[1],
                 "selected_patterns":"mean over 100 runs"})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "strategy_comparison.csv", index=False)

    # Greedy coverage curve for a transparent all-pattern comparison.
    curve_rows = []
    for name, order in [("AI_Ranked", ai_order), ("Basic_Binary", np.arange(len(patterns)))]:
        seen = np.zeros(matrix.shape[1], dtype=bool)
        for k, idx in enumerate(order, start=1):
            seen |= matrix[idx].astype(bool)
            curve_rows.append({"strategy":name, "patterns_applied":k,
                               "faults_detected":int(seen.sum()),
                               "fault_coverage":float(seen.mean())})
    pd.DataFrame(curve_rows).to_csv(RESULTS / "coverage_curve.csv", index=False)

    summary = {
        "circuit": "4-input, 3-output combinational logic network",
        "patterns_exhaustively_generated": len(patterns),
        "target_faults": len(faults),
        "fault_model": "single stuck-at-0 and stuck-at-1 on 7 internal/output nodes",
        "target_coverage": args.target_coverage,
        "cv_mae_mean": float(-cv_scores.mean()),
        "cv_mae_std": float(cv_scores.std()),
        "ai_selection": rows[0],
        "basic_selection": rows[1],
        "random_mean_100_runs": rows[2],
        "feature_importances": {name: float(value) for name,value in zip(X.columns, model.feature_importances_)},
        "limitations": "Toy combinational benchmark; results are educational and should not be generalized to industrial ATPG."
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("Project run completed.")
    print(comparison.to_string(index=False))
    print(f"Cross-validation MAE: {-cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Files written under: {ROOT}")
if __name__ == "__main__":
    main()
