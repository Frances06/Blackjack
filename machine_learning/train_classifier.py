"""Compare classifiers for predicting the best action from player-ev.json.

Run from the repo root:
    python3 -m machine_learning.train_classifier

For each model, 5-fold stratified cross-validation reports
    accuracy  -- share of decisions where the predicted action is the best one
    regret    -- mean EV lost per decision (cents per dollar bet) by the
                 model's mistakes; a near-miss on a close decision costs
                 almost nothing, a bad mistake costs a lot, so this is the
                 metric that matters for playing.
It also tunes the decision tree's depth, tests generalisation to an unseen
deck count, and writes the readable tree rules. Results go to
machine_learning/results/.
"""
import csv
import os
from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from machine_learning.classifier import (FEATURES, TREE_PARAMS, dataset_xy,
                                         ev_regret, load_rows)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def models():
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "Majority class (baseline)": DummyClassifier(strategy="most_frequent"),
        "Decision tree": DecisionTreeClassifier(**TREE_PARAMS),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=0),
        "k-nearest neighbours (k=5)": scaled(KNeighborsClassifier(5)),
        "Logistic regression": scaled(LogisticRegression(max_iter=2000)),
        "Neural network (MLP 64-64)": scaled(MLPClassifier((64, 64), max_iter=2000, random_state=0)),
    }


def folds(y, k=5):
    counts = Counter(y)
    if min(counts.values()) >= k:
        return StratifiedKFold(k, shuffle=True, random_state=0)
    return KFold(k, shuffle=True, random_state=0)


def cross_validate(model, rows, X, y, k=5):
    acc, reg = [], []
    for train, test in folds(y, k).split(X, y):
        model.fit([X[i] for i in train], [y[i] for i in train])
        pred = model.predict([X[i] for i in test])
        acc.append(sum(p == y[i] for p, i in zip(pred, test)) / len(test))
        reg.append(ev_regret([rows[i] for i in test], pred))
    return sum(acc) / k, sum(reg) / k


def write_csv(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def main():
    rows = load_rows()
    X, y = dataset_xy(rows)
    print(f"{len(rows)} decisions; label counts: {dict(Counter(y))}")
    print(f"Features: {', '.join(FEATURES)}\n")

    print("=== 5-fold cross-validation ===")
    print(f"{'Model':<28}{'Accuracy':>10}{'Regret (c/$)':>14}")
    table = []
    for name, m in models().items():
        acc, reg = cross_validate(m, rows, X, y)
        table.append((name, f"{acc:.4f}", f"{reg:.3f}"))
        print(f"{name:<28}{acc:>10.1%}{reg:>14.3f}")
    write_csv("classifier_comparison.csv", ["model", "cv_accuracy", "cv_regret_cents"], table)

    print("\n=== Decision tree depth (5-fold CV) ===")
    depth_rows = []
    for d in (2, 3, 4, 6, 8, 10, 12, 15, None):
        acc, reg = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=0), rows, X, y)
        depth_rows.append((d or "none", f"{acc:.4f}", f"{reg:.3f}"))
        print(f"max_depth={str(d):<6} accuracy {acc:6.1%}   regret {reg:6.3f} c/$")
    write_csv("tree_depth.csv", ["max_depth", "cv_accuracy", "cv_regret_cents"], depth_rows)

    decks = sorted({int(r["decks"]) for r in rows})
    if len(decks) > 1:
        print("\n=== Generalisation: train without one deck count, test on it ===")
        gen = []
        for d in decks:
            tr = [i for i, r in enumerate(rows) if int(r["decks"]) != d]
            te = [i for i, r in enumerate(rows) if int(r["decks"]) == d]
            m = DecisionTreeClassifier(**TREE_PARAMS).fit([X[i] for i in tr], [y[i] for i in tr])
            pred = m.predict([X[i] for i in te])
            acc = sum(p == y[i] for p, i in zip(pred, te)) / len(te)
            reg = ev_regret([rows[i] for i in te], pred)
            gen.append((d, f"{acc:.4f}", f"{reg:.3f}"))
            print(f"held-out {d} deck(s): accuracy {acc:6.1%}   regret {reg:6.3f} c/$")
        write_csv("tree_generalisation.csv", ["held_out_decks", "accuracy", "regret_cents"], gen)

    tree = DecisionTreeClassifier(**TREE_PARAMS).fit(X, y)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "tree_rules.txt"), "w") as f:
        f.write(export_text(tree, feature_names=FEATURES))
    print("\nFeature importances (decision tree):")
    for name, imp in sorted(zip(FEATURES, tree.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<14}{imp:.3f}")
    print(f"\nWrote CSVs and tree_rules.txt to {RESULTS}")


if __name__ == "__main__":
    main()
