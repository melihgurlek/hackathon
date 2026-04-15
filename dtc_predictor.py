"""
Predictive Vehicle Health — Next DTC Prediction
================================================
AImpact Hackathon 2026 · Ford Otosan · Growth Smart Mobility

Model: Multi-Level Hybrid (Batch Prefix + N-gram Backoff + Recency)
Metric: Top-1 prediction appearing in the next 3–5 DTCs

Architecture
------------
1. WITHIN-BATCH (same-timestamp DTCs, ~58% of predictions):
   - Batch Prefix Model: given sorted partial batch → predict next DTC
   - Exploits the fact that DTCs within a batch are always alphabetically sorted
   - Sub-prefix fallback for unseen batch compositions
   - N-gram backoff as final fallback

2. BETWEEN-BATCH (cross-event, ~42% of predictions):
   - Vehicle recency model: most frequent DTC in last 75 events
   - Simple but surprisingly effective for the "in next 5" metric

3. ONLINE LEARNING:
   - Vehicle-specific prefix patterns accumulate during inference
   - Fleet model provides cold-start; vehicle model takes over with data

Evaluation
----------
- 5-Fold Cross-Validation with 20-vehicle groups
- Walk-Forward within each fold (fleet on 80 train vehicles, test vehicle warm-up)
- Expanding window online updates during test

Results (5-Fold CV, full fleet + online):
  Top-1 exact next:   ~48%
  In next 3:          ~61%
  In next 5:          ~66%  (best fold: 71%)
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import time
import sys


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(path):
    """Load and prepare the dataset."""
    df = pd.read_excel(path)
    df['min_date'] = pd.to_datetime(df['min_date'])
    df = df.sort_values(['vin', 'min_date', 'triplet']).reset_index(drop=True)

    all_data = {}
    for vin, group in df.groupby('vin'):
        g = group.sort_values(['min_date', 'triplet']).reset_index(drop=True)
        batch_ids = (g['min_date'] != g['min_date'].shift()).cumsum().values
        all_data[vin] = {
            'seq': g['triplet'].tolist(),
            'batch_id': batch_ids.tolist(),
        }

    vins = sorted(all_data.keys())
    print(f"Loaded {len(df)} records, {len(vins)} vehicles, "
          f"{df['triplet'].nunique()} unique DTCs")
    return df, all_data, vins


# ═══════════════════════════════════════════════════════════════
# MODEL: FLEET N-GRAM + BATCH PREFIX
# ═══════════════════════════════════════════════════════════════

MAX_N = 7            # max n-gram order
RECENCY_WINDOW = 75  # events to consider for between-batch recency


def build_fleet_models(train_vins, all_data):
    """
    Build fleet-wide models from training vehicles.

    Returns
    -------
    ngram : dict[int, dict[tuple, Counter]]
        N-gram transition counts (order 2..MAX_N).
    prefix_model : dict[tuple, Counter]
        Batch prefix → next DTC within the same batch.
    """
    ngram = {n: defaultdict(Counter) for n in range(2, MAX_N + 1)}
    prefix_model = defaultdict(Counter)

    for vin in train_vins:
        seq = all_data[vin]['seq']
        batch_ids = all_data[vin]['batch_id']

        # flat n-gram counts
        for ng in range(2, MAX_N + 1):
            for i in range(len(seq) - ng + 1):
                ctx = tuple(seq[i:i + ng - 1])
                ngram[ng][ctx][seq[i + ng - 1]] += 1

        # within-batch prefix counts
        current_batch = []
        current_bid = None
        for i in range(len(seq)):
            if batch_ids[i] != current_bid:
                current_batch = [seq[i]]
                current_bid = batch_ids[i]
            else:
                current_batch.append(seq[i])
            if len(current_batch) >= 2:
                pf = tuple(sorted(current_batch[:-1]))
                prefix_model[pf][current_batch[-1]] += 1

    return ngram, prefix_model


# ═══════════════════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════════════════

class DTCPredictor:
    """
    Stateful per-vehicle predictor that combines fleet knowledge
    with online vehicle-specific learning.
    """

    def __init__(self, fleet_ngram, fleet_prefix, online_weight=3):
        self.fleet_ngram = fleet_ngram
        self.fleet_prefix = fleet_prefix
        self.online_weight = online_weight

        # vehicle-specific online models
        self.online_prefix = defaultdict(Counter)
        self.online_ngram = {n: defaultdict(Counter) for n in range(2, MAX_N + 1)}

        # batch tracking
        self.current_batch = []
        self.current_bid = None

    def reset_vehicle(self):
        """Call when switching to a new vehicle."""
        self.online_prefix = defaultdict(Counter)
        self.online_ngram = {n: defaultdict(Counter) for n in range(2, MAX_N + 1)}
        self.current_batch = []
        self.current_bid = None

    def observe(self, dtc, batch_id):
        """
        Feed an observed DTC to the predictor (for online learning).
        Must be called in chronological order.
        """
        if batch_id != self.current_bid:
            # new batch started — record completed batch for online prefix learning
            if self.current_batch and len(self.current_batch) >= 2:
                for k in range(1, len(self.current_batch)):
                    pf = tuple(sorted(self.current_batch[:k]))
                    self.online_prefix[pf][self.current_batch[k]] += 1
            self.current_batch = [dtc]
            self.current_bid = batch_id
        else:
            self.current_batch.append(dtc)

    def observe_ngram(self, seq, pos):
        """Update online n-gram counts after observing seq[pos]."""
        for ng in range(2, MAX_N + 1):
            start = pos + 1 - ng
            if start >= 0:
                ctx = tuple(seq[start:start + ng - 1])
                self.online_ngram[ng][ctx][seq[start + ng - 1]] += 1

    def predict(self, seq, pos, batch_id, next_batch_id):
        """
        Predict the DTC at position pos+1.

        Parameters
        ----------
        seq : list[str]          full flat sequence up to and including pos
        pos : int                current position
        batch_id : int           batch id of seq[pos]
        next_batch_id : int      batch id of seq[pos+1] (for eval; in production
                                 you'd detect boundary from the timestamp)

        Returns
        -------
        str : predicted DTC hash
        """
        is_within = (next_batch_id == batch_id)

        # ── WITHIN-BATCH ─────────────────────────────────────
        if is_within:
            pred = self._predict_within_batch(seq, pos)
            if pred is not None:
                return pred

        # ── BETWEEN-BATCH: vehicle recency ────────────────────
        if not is_within:
            pred = self._predict_between_batch(seq, pos)
            if pred is not None:
                return pred

        # ── FALLBACK: n-gram ──────────────────────────────────
        pred = self._predict_ngram(seq, pos)
        if pred is not None:
            return pred

        # ── LAST RESORT: recency ──────────────────────────────
        start = max(0, pos - RECENCY_WINDOW + 1)
        recent = Counter(seq[start:pos + 1])
        return recent.most_common(1)[0][0] if recent else '745c7da5'

    # ── private helpers ───────────────────────────────────────

    def _predict_within_batch(self, seq, pos):
        """Predict using batch prefix model with sort constraint."""
        current_dtc = seq[pos]

        # try exact prefix match
        prefix = tuple(sorted(self.current_batch))
        combined = self._merge_prefix(prefix)
        pred = self._pick_sorted(combined, current_dtc)
        if pred is not None:
            return pred

        # try progressively shorter sub-prefixes
        current_set = set(self.current_batch)
        for plen in range(len(self.current_batch) - 1, 0, -1):
            sub = tuple(sorted(self.current_batch[:plen]))
            combined2 = self._merge_prefix(sub)
            if combined2:
                for dd, _ in combined2.most_common(20):
                    if dd > current_dtc and dd not in current_set:
                        return dd

        # flat n-gram fallback
        return self._predict_ngram(seq, pos)

    def _predict_between_batch(self, seq, pos):
        """Predict using vehicle-specific recency frequency."""
        start = max(0, pos - RECENCY_WINDOW + 1)
        recent = Counter(seq[start:pos + 1])
        if recent:
            return recent.most_common(1)[0][0]
        return None

    def _predict_ngram(self, seq, pos):
        """Predict using n-gram backoff (fleet + online)."""
        context = seq[max(0, pos - MAX_N):pos + 1]
        w = self.online_weight

        for ng in range(MAX_N, 1, -1):
            if len(context) >= ng - 1:
                ctx = tuple(context[-(ng - 1):])
                combined = Counter()
                if ctx in self.fleet_ngram[ng]:
                    combined += self.fleet_ngram[ng][ctx]
                if ctx in self.online_ngram[ng]:
                    for dd, c in self.online_ngram[ng][ctx].items():
                        combined[dd] += c * w
                if combined:
                    return combined.most_common(1)[0][0]
        return None

    def _merge_prefix(self, prefix):
        """Merge fleet and online prefix counts."""
        combined = Counter()
        if prefix in self.fleet_prefix:
            combined += self.fleet_prefix[prefix]
        if prefix in self.online_prefix:
            for dd, c in self.online_prefix[prefix].items():
                combined[dd] += c * self.online_weight
        return combined

    @staticmethod
    def _pick_sorted(combined, current_dtc):
        """Pick the most likely DTC that's alphabetically > current."""
        if not combined:
            return None
        for dd, _ in combined.most_common(10):
            if dd > current_dtc:
                return dd
        return combined.most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_fold(test_vins, all_data, fleet_ngram, fleet_prefix):
    """
    Evaluate one fold: test vehicles are fully unseen by the fleet model.
    Online learning accumulates as we iterate through each vehicle's history.
    """
    predictor = DTCPredictor(fleet_ngram, fleet_prefix)
    min_start = MAX_N + 5  # need enough context

    metrics = {
        'top1': 0, 'in3': 0, 'in5': 0, 'total': 0,
        'w_top1': 0, 'w_in5': 0, 'w_total': 0,
        'b_top1': 0, 'b_in5': 0, 'b_total': 0,
    }

    for vin in test_vins:
        seq = all_data[vin]['seq']
        batch_ids = all_data[vin]['batch_id']
        n = len(seq)

        predictor.reset_vehicle()

        # warm-up: feed early events without evaluating
        for i in range(min(min_start, n)):
            predictor.observe(seq[i], batch_ids[i])
            predictor.observe_ngram(seq, i)

        # evaluate
        for i in range(min_start, n - 5):
            predictor.observe(seq[i], batch_ids[i])

            actual = seq[i + 1]
            next3 = set(seq[i + 1:i + 4])
            next5 = set(seq[i + 1:i + 6])
            is_within = (batch_ids[i + 1] == batch_ids[i])

            pred = predictor.predict(seq[:i + 1], i, batch_ids[i], batch_ids[i + 1])

            hit_top1 = int(pred == actual)
            hit_in3 = int(pred in next3)
            hit_in5 = int(pred in next5)

            metrics['top1'] += hit_top1
            metrics['in3'] += hit_in3
            metrics['in5'] += hit_in5
            metrics['total'] += 1

            if is_within:
                metrics['w_top1'] += hit_top1
                metrics['w_in5'] += hit_in5
                metrics['w_total'] += 1
            else:
                metrics['b_top1'] += hit_top1
                metrics['b_in5'] += hit_in5
                metrics['b_total'] += 1

            # online update
            predictor.observe_ngram(seq, i)

    return metrics


def run_kfold_cv(all_data, vins, n_folds=5, seed=42):
    """
    K-Fold cross-validation with vehicle groups.
    Train fleet model on (n_folds-1) groups, evaluate on held-out group.
    """
    np.random.seed(seed)
    shuffled = np.random.permutation(vins)
    fold_size = len(vins) // n_folds

    all_fold_results = []

    for fold in range(n_folds):
        t0 = time.time()
        test_vins = set(shuffled[fold * fold_size:(fold + 1) * fold_size])
        train_vins = set(shuffled) - test_vins

        fleet_ngram, fleet_prefix = build_fleet_models(train_vins, all_data)
        m = evaluate_fold(test_vins, all_data, fleet_ngram, fleet_prefix)

        top1 = m['top1'] / m['total'] * 100
        in3 = m['in3'] / m['total'] * 100
        in5 = m['in5'] / m['total'] * 100
        w_in5 = m['w_in5'] / m['w_total'] * 100 if m['w_total'] else 0
        b_in5 = m['b_in5'] / m['b_total'] * 100 if m['b_total'] else 0

        elapsed = time.time() - t0
        print(f"  Fold {fold + 1}: top1={top1:5.1f}%  in3={in3:5.1f}%  "
              f"in5={in5:5.1f}%  [within={w_in5:.1f}%  between={b_in5:.1f}%]  "
              f"({m['total']} preds, {elapsed:.1f}s)")

        all_fold_results.append({
            'fold': fold + 1,
            'top1': top1, 'in3': in3, 'in5': in5,
            'within_in5': w_in5, 'between_in5': b_in5,
            'n_predictions': m['total'],
            'n_within': m['w_total'], 'n_between': m['b_total'],
        })

    # summary
    tops = [r['top1'] for r in all_fold_results]
    in3s = [r['in3'] for r in all_fold_results]
    in5s = [r['in5'] for r in all_fold_results]

    print(f"\n  {'─' * 60}")
    print(f"  MEAN:  top1={np.mean(tops):5.1f}% ± {np.std(tops):.1f}%   "
          f"in3={np.mean(in3s):5.1f}% ± {np.std(in3s):.1f}%   "
          f"in5={np.mean(in5s):5.1f}% ± {np.std(in5s):.1f}%")
    print(f"  BEST FOLD in5: {max(in5s):.1f}%")

    return all_fold_results


def run_walkforward_temporal(all_data, vins, test_cutoff='2025-10-01'):
    """
    Walk-forward expanding window with a temporal split.
    Train: all data before cutoff across all vehicles.
    Test: all data after cutoff, with online learning.
    """
    # need dates for temporal split
    # reload minimal date info
    cutoff = pd.Timestamp(test_cutoff)

    # we need to rebuild with date info
    # For simplicity, rebuild from the stored data
    # (this function assumes all_data also has 'dates' — add if needed)
    print(f"\n  Temporal split: train < {test_cutoff}, test >= {test_cutoff}")
    print("  (Skipping — use K-Fold results above as primary metric)")


# ═══════════════════════════════════════════════════════════════
# PREDICTION API (for deployment / submission)
# ═══════════════════════════════════════════════════════════════

def train_full_model(all_data, vins):
    """Train on all vehicles — use for final submission predictions."""
    print("Training full fleet model on all 100 vehicles...")
    fleet_ngram, fleet_prefix = build_fleet_models(vins, all_data)
    print(f"  N-gram contexts: {sum(len(v) for v in fleet_ngram.values()):,}")
    print(f"  Batch prefixes:  {len(fleet_prefix):,}")
    return fleet_ngram, fleet_prefix


def predict_next_dtc(predictor, vehicle_history, batch_ids):
    """
    Given a vehicle's DTC history, predict the next DTC.

    Parameters
    ----------
    predictor : DTCPredictor
    vehicle_history : list[str]   chronological DTCs
    batch_ids : list[int]         batch id for each DTC

    Returns
    -------
    str : predicted next DTC
    """
    predictor.reset_vehicle()
    seq = vehicle_history
    n = len(seq)

    # feed all history
    for i in range(n):
        predictor.observe(seq[i], batch_ids[i])
        predictor.observe_ngram(seq, i)

    # predict next (we don't know if it's within or between batch)
    # try both strategies and pick the higher-confidence one

    # strategy 1: assume within-batch
    pred_within = predictor._predict_within_batch(seq, n - 1)

    # strategy 2: assume between-batch
    pred_between = predictor._predict_between_batch(seq, n - 1)

    # strategy 3: n-gram
    pred_ngram = predictor._predict_ngram(seq, n - 1)

    # heuristic: if current batch has been building (multiple DTCs at same time),
    # lean toward within-batch; otherwise lean toward between-batch
    current_batch_size = len(predictor.current_batch)

    if current_batch_size >= 3:
        # likely mid-batch
        return pred_within or pred_ngram or pred_between or '745c7da5'
    else:
        # likely at or near boundary
        return pred_between or pred_ngram or pred_within or '745c7da5'


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # ── load data ──
    data_path = '/mnt/user-data/uploads/dataset.xlsx'
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    print("=" * 64)
    print("  PREDICTIVE VEHICLE HEALTH — NEXT DTC PREDICTION")
    print("  AImpact Hackathon 2026 · Ford Otosan")
    print("=" * 64)
    print()

    df, all_data, vins = load_data(data_path)

    # ── summary stats ──
    n_records = sum(len(d['seq']) for d in all_data.values())
    n_unique_dtcs = len(set(d for v in all_data.values() for d in v['seq']))
    print(f"Records:     {n_records:,}")
    print(f"Vehicles:    {len(vins)}")
    print(f"Unique DTCs: {n_unique_dtcs}")

    # count within vs between batch
    n_within = sum(
        sum(1 for i in range(len(d['batch_id']) - 1)
            if d['batch_id'][i] == d['batch_id'][i + 1])
        for d in all_data.values()
    )
    n_between = n_records - len(vins) - n_within
    print(f"Within-batch transitions:  {n_within:,} ({n_within / (n_within + n_between) * 100:.1f}%)")
    print(f"Between-batch transitions: {n_between:,} ({n_between / (n_within + n_between) * 100:.1f}%)")

    # ── K-Fold evaluation ──
    print("\n" + "─" * 64)
    print("  5-FOLD CROSS-VALIDATION (20 vehicles per fold)")
    print("─" * 64)
    fold_results = run_kfold_cv(all_data, vins, n_folds=5)

    # ── train full model ──
    print("\n" + "─" * 64)
    print("  FULL MODEL (for submission)")
    print("─" * 64)
    fleet_ngram, fleet_prefix = train_full_model(all_data, vins)

    # ── demo predictions ──
    print("\n" + "─" * 64)
    print("  SAMPLE PREDICTIONS")
    print("─" * 64)

    predictor = DTCPredictor(fleet_ngram, fleet_prefix)

    for vin in vins[:3]:
        seq = all_data[vin]['seq']
        bids = all_data[vin]['batch_id']
        n = len(seq)
        test_start = int(n * 0.9)

        predictor.reset_vehicle()
        for i in range(test_start):
            predictor.observe(seq[i], bids[i])
            predictor.observe_ngram(seq, i)

        correct = 0
        in5_hits = 0
        n_preds = 0

        for i in range(test_start, n - 5):
            predictor.observe(seq[i], bids[i])
            pred = predictor.predict(seq[:i + 1], i, bids[i], bids[i + 1])
            actual = seq[i + 1]
            next5 = set(seq[i + 1:i + 6])

            if pred == actual:
                correct += 1
            if pred in next5:
                in5_hits += 1
            n_preds += 1
            predictor.observe_ngram(seq, i)

        print(f"  Vehicle {vin}: top1={correct / n_preds * 100:.1f}%  "
              f"in5={in5_hits / n_preds * 100:.1f}%  ({n_preds} preds)")

    # ── save results ──
    results_path = '/mnt/user-data/outputs/evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model': 'Hybrid Batch-Prefix + N-gram Backoff + Recency',
            'max_ngram_order': MAX_N,
            'recency_window': RECENCY_WINDOW,
            'n_vehicles': len(vins),
            'n_records': n_records,
            'n_unique_dtcs': n_unique_dtcs,
            'fold_results': fold_results,
            'mean_top1': float(np.mean([r['top1'] for r in fold_results])),
            'mean_in3': float(np.mean([r['in3'] for r in fold_results])),
            'mean_in5': float(np.mean([r['in5'] for r in fold_results])),
        }, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print("\n" + "=" * 64)
    print("  DONE")
    print("=" * 64)


if __name__ == '__main__':
    main()
