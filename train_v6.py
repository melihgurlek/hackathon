# ╔══════════════════════════════════════════════════════════════════╗
# ║  Ford Otosan DTC Prediction  —  train_v6.py                     ║
# ║  Changes from v5:                                               ║
# ║  1. LSTM dropped from ensemble (TX alone beats TX+LSTM)         ║
# ║  2. Transformer: d_model 128→256, layers 3→4, heads 4→8        ║
# ║  3. Per-DTC recency feature — "when did THIS code last fire?"   ║
# ║  4. Label smoothing (ε=0.1) — better calibration               ║
# ║  5. Linear LR warmup — more stable early training               ║
# ║  6. LSTM still trained and saved for demo_export.py             ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# COLAB:
#   from google.colab import drive; drive.mount('/content/drive')
#   !pip install lightgbm openpyxl -q
#   !cp "/content/drive/MyDrive/dataset.xlsx" .

import os, json, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────
FILE_PATH  = "dataset.xlsx"
SAVE_DIR   = "/content/drive/MyDrive/ford_dtc"
SEQ_LEN    = 20
TOP_K      = 5
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
BATCH_SIZE = 256          # reduced: bigger model needs smaller batch to fit VRAM
MAX_SET    = 10

FOCAL_GAMMA  = 2.0
FOCAL_POS_W  = 20.0
LABEL_SMOOTH = 0.1        # smooth targets: 1→0.9, 0→0.1

# Transformer — bigger
TX_D_MODEL  = 256         # was 128
TX_NHEAD    = 8           # was 4  (must divide d_model)
TX_LAYERS   = 4           # was 3
TX_DROPOUT  = 0.1
TX_EPOCHS   = 40
TX_PATIENCE = 7
TX_LR       = 2e-4        # slightly lower for bigger model
TX_WARMUP   = 3           # warmup epochs

# LSTM — kept for demo_export, not in ensemble
LSTM_D_EMBED  = 128
LSTM_HIDDEN   = 256
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.2
LSTM_EPOCHS   = 25
LSTM_PATIENCE = 5
LSTM_LR       = 3e-4

LGBM_TOP_N  = 150

os.makedirs(SAVE_DIR, exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Check VRAM — if < 8GB, revert to d_model=128
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb < 8:
        TX_D_MODEL = 128
        TX_NHEAD   = 4
        TX_LAYERS  = 3
        BATCH_SIZE = 512
        print("  < 8GB VRAM detected — reverting to d_model=128")


# ═══════════════════════════════════════════════════════════════════════
# LOSS — focal + label smoothing
# ═══════════════════════════════════════════════════════════════════════
class FocalSmoothLoss(nn.Module):
    """
    Focal loss + label smoothing combined.
    Label smoothing: replace hard 0/1 targets with ε/2 and 1-ε/2.
    This prevents the model from being overconfident and improves calibration.
    Focal: downweights easy negatives so model focuses on hard positives.
    """
    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=FOCAL_POS_W, smooth=LABEL_SMOOTH):
        super().__init__()
        self.gamma  = gamma
        self.pw     = pos_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply label smoothing
        targets_s = targets * (1 - self.smooth) + (1 - targets) * self.smooth / 2
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_s,
            pos_weight=torch.tensor(self.pw, device=logits.device),
            reduction="none"
        )
        p     = torch.sigmoid(logits)
        p_t   = p * targets + (1 - p) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════
def load_and_build_events(path):
    print("Loading data...")
    df = pd.read_excel(path)
    df["min_date"] = pd.to_datetime(df["min_date"])
    df["max_date"] = pd.to_datetime(df["max_date"])
    df = df.sort_values(["vin", "min_date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows | {df['vin'].nunique()} VINs | {df['triplet'].nunique()} DTCs")
    ev = (
        df.groupby(["vin", "min_date"], sort=False)
        .agg(dtc_set=("triplet", frozenset),
             first_odo=("first_odometer", "min"),
             max_odo=("last_odometer", "max"),
             max_date=("max_date", "max"))
        .reset_index()
    )
    ev["fault_dur_h"] = (ev["max_date"] - ev["min_date"]).dt.total_seconds() / 3600
    ev = ev.sort_values(["vin", "min_date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows → {len(ev):,} events")
    return df, ev


def engineer_features(ev):
    print("Engineering features...")
    g = ev.groupby("vin")
    ev["prev_dtc_set_1"] = g["dtc_set"].shift(1)
    ev["prev_dtc_set_2"] = g["dtc_set"].shift(2)
    ev["prev_min_date"]  = g["min_date"].shift(1)
    ev["prev_odo"]       = g["first_odo"].shift(1)
    ev["time_gap_h"]     = (ev["min_date"] - ev["prev_min_date"]).dt.total_seconds() / 3600
    ev["log_time_gap"]   = np.log1p(ev["time_gap_h"].fillna(0))
    ev["odo_gap"]        = (ev["first_odo"] - ev["prev_odo"].fillna(ev["first_odo"])).clip(lower=0)
    ev["odo_drift"]      = ev["max_odo"] - ev["first_odo"]
    ev["event_size"]     = ev["dtc_set"].apply(len)
    ev["rolling_events"] = g.cumcount().clip(upper=20) / 20.0
    ev["next_dtc_set"]   = g["dtc_set"].shift(-1)
    ev = ev.dropna(subset=["prev_dtc_set_1", "next_dtc_set"]).copy()
    print(f"  {len(ev):,} usable events")
    return ev


def temporal_split(ev):
    tr, va, te = [], [], []
    for _, g in ev.groupby("vin"):
        n = len(g); a = int(n*TRAIN_FRAC); b = int(n*(TRAIN_FRAC+VAL_FRAC))
        tr.append(g.iloc[:a]); va.append(g.iloc[a:b]); te.append(g.iloc[b:])
    tr = pd.concat(tr).reset_index(drop=True)
    va = pd.concat(va).reset_index(drop=True)
    te = pd.concat(te).reset_index(drop=True)
    print(f"Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")
    return tr, va, te


# ═══════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════
class DTCVocab:
    def __init__(self):
        self.le = LabelEncoder()

    def fit(self, train_ev):
        dtcs = set()
        for s in train_ev["dtc_set"]: dtcs.update(s)
        self.le.fit(sorted(dtcs))
        self.size = len(self.le.classes_)
        self._enc = {d: int(i)+1 for i, d in enumerate(self.le.classes_)}
        print(f"  Vocab: {self.size} DTCs")
        return self

    def encode(self, d):     return self._enc.get(d, -1)
    def encode_set(self, s): return [e for d in s if (e := self._enc.get(d, -1)) > 0]
    def decode(self, i):     return self.le.inverse_transform([i-1])[0]
    def save(self, p):       joblib.dump(self, p)


# ═══════════════════════════════════════════════════════════════════════
# DATASET — with per-DTC recency features
# ═══════════════════════════════════════════════════════════════════════
class EventSeqDataset(Dataset):
    """
    Per-DTC recency: for each event in the window, how many events ago
    did each DTC in the CURRENT event last appear? Capped at SEQ_LEN,
    normalized 0-1. This is a unique signal — captures cyclic patterns
    (e.g. 7ce3a42b fires every 3 events) that the sequence embedding alone
    can't easily learn.

    Shape: each example has an extra (SEQ_LEN,) float32 array 'recency'
    = mean events-since-last-seen for the DTCs in that position's set.
    """
    def __init__(self, ev, vocab, seq_len=SEQ_LEN, max_set=MAX_SET):
        PAD = [0] * max_set
        examples = []

        for _, g in ev.groupby("vin"):
            g = g.reset_index(drop=True)
            n = len(g)

            enc_sets  = []
            for s in g["dtc_set"]:
                e = [v for d in s if (v := vocab._enc.get(d, -1)) > 0]
                e = e[:max_set] + [0]*(max_set - len(e[:max_set]))
                enc_sets.append(e)

            next_enc  = [vocab.encode_set(s) for s in g["next_dtc_set"]]
            tgaps     = g["log_time_gap"].fillna(0).to_numpy(np.float32)
            odos      = (g["first_odo"].fillna(0)/100.0).to_numpy(np.float32)

            # Build last-seen index for each DTC (per-VIN history)
            last_seen = {}   # dtc_enc -> most recent event index

            for i in range(n):
                # Update last_seen with current event's DTCs
                for dtc_enc in enc_sets[i]:
                    if dtc_enc > 0:
                        last_seen[dtc_enc] = i

                ne = next_enc[i]
                if not ne: continue

                target = np.zeros(vocab.size + 1, dtype=np.float32)
                for idx in ne:
                    if 0 < idx <= vocab.size: target[idx] = 1.0

                start = max(0, i - seq_len + 1)
                pad_n = seq_len - (i + 1 - start)

                # Recency: for each window position, mean events-since-last-seen
                # for the DTCs in that position. 0 = just fired, 1 = seq_len+ ago
                recency = []
                for j in range(start, i+1):
                    dtcs_here = [d for d in enc_sets[j] if d > 0]
                    if dtcs_here:
                        ages = [min(i - last_seen.get(d, 0), seq_len) / seq_len
                                for d in dtcs_here]
                        recency.append(float(np.mean(ages)))
                    else:
                        recency.append(1.0)

                examples.append((
                    np.array([PAD]*pad_n + enc_sets[start:i+1],  dtype=np.int64),
                    np.array([0.]*pad_n  + tgaps[start:i+1].tolist(), dtype=np.float32),
                    np.array([0.]*pad_n  + odos[start:i+1].tolist(),  dtype=np.float32),
                    np.array([1.]*pad_n  + recency,                    dtype=np.float32),
                    target,
                ))

        self.dtc_seqs  = np.stack([e[0] for e in examples])
        self.time_gaps = np.stack([e[1] for e in examples])
        self.odo_vals  = np.stack([e[2] for e in examples])
        self.recency   = np.stack([e[3] for e in examples])
        self.targets   = np.stack([e[4] for e in examples])
        print(f"    {len(examples):,} sequences")

    def __len__(self): return len(self.dtc_seqs)
    def __getitem__(self, i):
        return (torch.from_numpy(self.dtc_seqs[i]),
                torch.from_numpy(self.time_gaps[i]),
                torch.from_numpy(self.odo_vals[i]),
                torch.from_numpy(self.recency[i]),
                torch.from_numpy(self.targets[i]))


def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=(DEVICE=="cuda"),
                      persistent_workers=True)


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════
def top_k_acc(true_sets, pred_lists, k=TOP_K):
    return sum(1 for t, p in zip(true_sets, pred_lists) if set(p[:k]) & t) / len(true_sets)

def prec_at_k(true_sets, pred_lists, k=TOP_K):
    return float(np.mean([len(set(p[:k]) & t) / k for t, p in zip(true_sets, pred_lists)]))

@torch.no_grad()
def get_probs_and_targets(model, loader):
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        *inputs, tgt = batch
        logits = model(*[x.to(DEVICE) for x in inputs])
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_t.append(tgt.numpy())
    return np.vstack(all_p), np.vstack(all_t)

def acc_from_probs(probs, targets, k=TOP_K):
    top_p     = np.argsort(-probs, axis=1)[:, :k]
    true_sets = [set(np.where(t > 0)[0]) - {0} for t in targets]
    return top_k_acc(true_sets, [list(r) for r in top_p], k)


# ═══════════════════════════════════════════════════════════════════════
# TRANSFORMER — bigger, takes recency as 3rd continuous feature
# ═══════════════════════════════════════════════════════════════════════
class DTCTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=TX_D_MODEL, nhead=TX_NHEAD,
                 num_layers=TX_LAYERS, seq_len=SEQ_LEN, dropout=TX_DROPOUT):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size + 2, d_model, padding_idx=0)
        self.cont_proj   = nn.Linear(3, d_model)    # 3 cont features: tgap, odo, recency
        self.pos_embed   = nn.Embedding(seq_len + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout     = nn.Dropout(dropout)
        self.head        = nn.Linear(d_model, vocab_size + 1)

    def forward(self, dtc_seq, tg, od, recency):
        B, S, M = dtc_seq.shape
        tok     = self.token_embed(dtc_seq)
        mask_   = (dtc_seq != 0).unsqueeze(-1).float()
        event_e = (tok * mask_).sum(2) / mask_.sum(2).clamp(min=1)
        cont    = torch.stack([tg, od, recency], dim=-1)     # now 3 features
        pos     = torch.arange(S, device=dtc_seq.device).unsqueeze(0)
        x       = event_e + self.cont_proj(cont) + self.pos_embed(pos)
        x       = self.dropout(x)
        causal  = nn.Transformer.generate_square_subsequent_mask(S, device=dtc_seq.device)
        x       = self.transformer(x, mask=causal, is_causal=True)
        return self.head(x[:, -1, :])


# ═══════════════════════════════════════════════════════════════════════
# LSTM — also takes recency
# ═══════════════════════════════════════════════════════════════════════
class DTCLstm(nn.Module):
    def __init__(self, vocab_size, d_embed=LSTM_D_EMBED, hidden=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size + 2, d_embed, padding_idx=0)
        self.cont_proj   = nn.Linear(3, d_embed)    # 3 cont features
        self.lstm        = nn.LSTM(d_embed, hidden, num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.dropout     = nn.Dropout(dropout)
        self.head        = nn.Linear(hidden, vocab_size + 1)

    def forward(self, dtc_seq, tg, od, recency):
        B, S, M = dtc_seq.shape
        tok     = self.token_embed(dtc_seq)
        mask_   = (dtc_seq != 0).unsqueeze(-1).float()
        event_e = (tok * mask_).sum(2) / mask_.sum(2).clamp(min=1)
        cont    = torch.stack([tg, od, recency], dim=-1)
        x       = self.dropout(event_e + self.cont_proj(cont))
        out, _  = self.lstm(x)
        return self.head(out[:, -1, :])


# ═══════════════════════════════════════════════════════════════════════
# TRAINING — with linear warmup
# ═══════════════════════════════════════════════════════════════════════
class WarmupCosineScheduler:
    """Linear warmup for `warmup` epochs, then cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.opt     = optimizer
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.base_lr = base_lr
        self.epoch   = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            lr = self.base_lr * (self.epoch / self.warmup)
        else:
            progress = (self.epoch - self.warmup) / (self.total - self.warmup)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def train_model(model, tr_dl, va_dl, epochs, patience, lr, warmup,
                save_path, label):
    print(f"\n── Training {label} {'─'*(42-len(label))}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr/warmup, weight_decay=1e-4)
    sched = WarmupCosineScheduler(opt, warmup, epochs, lr)
    crit  = FocalSmoothLoss()
    best_val, patience_ctr = -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in tr_dl:
            *inputs, tgt = batch
            inputs = [x.to(DEVICE) for x in inputs]
            tgt    = tgt.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*inputs), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        cur_lr = sched.step()

        val_p, val_t = get_probs_and_targets(model, va_dl)
        val_acc = acc_from_probs(val_p, val_t)
        print(f"  Epoch {epoch:2d}/{epochs} | loss={total_loss/len(tr_dl):.4f} "
              f"| val_top{TOP_K}={val_acc:.4f} | lr={cur_lr:.2e} | {time.time()-t0:.0f}s")

        if val_acc > best_val:
            best_val, patience_ctr = val_acc, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"  Best val top-{TOP_K}: {best_val:.4f}")
    return model, best_val


# ═══════════════════════════════════════════════════════════════════════
# LIGHTGBM — standalone, not in ensemble
# ═══════════════════════════════════════════════════════════════════════
class LGBMPredictor:
    def __init__(self, vocab, top_n=LGBM_TOP_N):
        self.vocab    = vocab; self.top_n = top_n
        self.models   = {}; self.top_dtcs = []
        self.scaler   = MinMaxScaler()
        self.cont_cols = ["first_odo", "log_time_gap", "fault_dur_h",
                          "odo_gap", "odo_drift", "event_size", "rolling_events"]

    def _multihot(self, ev):
        d2c = {d: c for c, d in enumerate(self.top_dtcs)}
        mat = np.zeros((len(ev), self.top_n), dtype=np.float32)
        for i, s in enumerate(ev["prev_dtc_set_1"]):
            for d in s:
                enc = self.vocab._enc.get(d, -1)
                if enc in d2c: mat[i, d2c[enc]] = 1.0
        return mat

    def _build_X(self, ev, fit=False):
        cont = ev[self.cont_cols].fillna(0).values.astype(np.float32)
        cont = self.scaler.fit_transform(cont) if fit else self.scaler.transform(cont)
        return np.hstack([self._multihot(ev), cont])

    def fit(self, tr, va):
        import lightgbm as lgb
        print(f"\n── Training LightGBM (standalone) ───────────────────")
        freq = {}
        for s in tr["dtc_set"]:
            for d in s:
                e = self.vocab._enc.get(d, -1)
                if e > 0: freq[e] = freq.get(e, 0) + 1
        self.top_dtcs = [d for d,_ in sorted(freq.items(), key=lambda x:-x[1])][:self.top_n]
        X_tr = self._build_X(tr, fit=True); X_va = self._build_X(va)
        print(f"  X: {X_tr.shape}")
        for rank, dtc_idx in enumerate(self.top_dtcs):
            y_tr = np.array([int(dtc_idx in self.vocab.encode_set(s)) for s in tr["next_dtc_set"]], dtype=np.int8)
            y_va = np.array([int(dtc_idx in self.vocab.encode_set(s)) for s in va["next_dtc_set"]], dtype=np.int8)
            if y_tr.sum() < 5: continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                     min_child_samples=10, scale_pos_weight=15,
                                     random_state=42, verbose=-1, n_jobs=-1)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            self.models[dtc_idx] = clf
            if (rank+1) % 50 == 0: print(f"  {rank+1}/{self.top_n}")
        print(f"  Trained {len(self.models)} classifiers")
        return self

    def get_prob_vector(self, ev):
        X = self._build_X(ev)
        mat = np.zeros((len(ev), self.vocab.size+1), dtype=np.float32)
        for ci, dtc_idx in enumerate(self.top_dtcs):
            if dtc_idx in self.models and 0 < dtc_idx <= self.vocab.size:
                mat[:, dtc_idx] = self.models[dtc_idx].predict_proba(X)[:, 1]
        return mat

    def evaluate(self, ev, k=TOP_K):
        probs = self.get_prob_vector(ev); top_p = np.argsort(-probs, axis=1)[:, :k]
        true_sets = [set(self.vocab.encode_set(s)) for s in ev["next_dtc_set"]]
        return {"top_k_acc": top_k_acc(true_sets, [list(r) for r in top_p], k),
                "prec_at_k": prec_at_k(true_sets, [list(r) for r in top_p], k)}

    def save(self, p): joblib.dump(self, p)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    raw, ev    = load_and_build_events(FILE_PATH)
    ev         = engineer_features(ev)
    tr, va, te = temporal_split(ev)

    print("\nBuilding vocabulary...")
    vocab = DTCVocab().fit(tr)
    vocab.save(f"{SAVE_DIR}/dtc_vocab.pkl")

    print("\nBuilding datasets...")
    print("  Train:"); tr_ds = EventSeqDataset(tr, vocab)
    print("  Val:");   va_ds = EventSeqDataset(va, vocab)
    print("  Test:");  te_ds = EventSeqDataset(te, vocab)

    tr_dl = make_loader(tr_ds, shuffle=True)
    va_dl = make_loader(va_ds, shuffle=False)
    te_dl = make_loader(te_ds, shuffle=False)

    # ── Transformer ───────────────────────────────────────────────────
    tx = DTCTransformer(vocab.size).to(DEVICE)
    tx, tx_val = train_model(tx, tr_dl, va_dl, TX_EPOCHS, TX_PATIENCE,
                              TX_LR, TX_WARMUP,
                              f"{SAVE_DIR}/transformer_best.pt", "Transformer")
    torch.save(tx.state_dict(), f"{SAVE_DIR}/transformer_final.pt")

    # ── LSTM (saved for demo_export but not in ensemble) ─────────────
    lstm = DTCLstm(vocab.size).to(DEVICE)
    lstm, lstm_val = train_model(lstm, tr_dl, va_dl, LSTM_EPOCHS, LSTM_PATIENCE,
                                  LSTM_LR, 2,
                                  f"{SAVE_DIR}/lstm_best.pt", "LSTM")
    torch.save(lstm.state_dict(), f"{SAVE_DIR}/lstm_final.pt")

    # ── LightGBM ──────────────────────────────────────────────────────
    lgbm = LGBMPredictor(vocab).fit(tr, va)
    lgbm_val = lgbm.evaluate(va); lgbm_te = lgbm.evaluate(te)
    lgbm.save(f"{SAVE_DIR}/lgbm.pkl")

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n── Final test evaluation ─────────────────────────────────")
    tx_te_p,   te_t = get_probs_and_targets(tx,   te_dl)
    lstm_te_p, _    = get_probs_and_targets(lstm, te_dl)
    tx_te_acc   = acc_from_probs(tx_te_p,   te_t)
    lstm_te_acc = acc_from_probs(lstm_te_p, te_t)

    # TX-only is the ensemble (learned from v5 that LSTM hurts)
    # But try a light blend in case LSTM improved
    va_tx_p, va_t = get_probs_and_targets(tx,   va_dl)
    va_lstm_p, _  = get_probs_and_targets(lstm, va_dl)

    best_val_acc, best_w = 0.0, 1.0
    for w in np.arange(0.5, 1.01, 0.05):
        comb = w * va_tx_p + (1-w) * va_lstm_p
        comb[:, 0] = 0
        acc = acc_from_probs(comb, va_t)
        if acc > best_val_acc:
            best_val_acc, best_w = acc, round(w, 2)

    final_p = best_w * tx_te_p + (1-best_w) * lstm_te_p
    final_p[:, 0] = 0
    final_acc  = acc_from_probs(final_p, te_t)
    true_sets  = [set(np.where(t > 0)[0]) - {0} for t in te_t]
    pred_lists = [list(r) for r in np.argsort(-final_p, axis=1)[:, :TOP_K]]
    final_prec = prec_at_k(true_sets, pred_lists)

    print(f"\n{'='*54}")
    print(f"  Transformer   val={tx_val:.4f}   test={tx_te_acc:.4f}")
    print(f"  LSTM          val={lstm_val:.4f}   test={lstm_te_acc:.4f}")
    print(f"  LightGBM      val={lgbm_val['top_k_acc']:.4f}   test={lgbm_te['top_k_acc']:.4f}")
    print(f"  ──────────────────────────────────────────────────────")
    w_str = f"TX={best_w:.2f} LSTM={1-best_w:.2f}"
    print(f"  Ensemble ({w_str})  test={final_acc:.4f}  ← final")
    print(f"  prec@{TOP_K}:                           {final_prec:.4f}")
    print(f"{'='*54}")

    verdict = "TX alone" if best_w >= 0.95 else f"TX {best_w:.0%} + LSTM {1-best_w:.0%}"
    print(f"\nBest ensemble: {verdict}")

    meta = {
        "tx_val": tx_val, "tx_test": tx_te_acc,
        "lstm_val": lstm_val, "lstm_test": lstm_te_acc,
        "lgbm_val": lgbm_val["top_k_acc"], "lgbm_test": lgbm_te["top_k_acc"],
        "ensemble_test": final_acc, "ensemble_prec": final_prec,
        "weights": {"transformer": best_w, "lstm": round(1-best_w, 2)},
        "vocab_size": vocab.size, "seq_len": SEQ_LEN, "top_k": TOP_K,
        "d_model": TX_D_MODEL, "tx_layers": TX_LAYERS,
    }
    with open(f"{SAVE_DIR}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min | Saved to {SAVE_DIR}")
