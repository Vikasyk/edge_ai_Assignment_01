"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 12                     # our final model (the defaults below)
    python train.py --team 12 --hidden 32,32      # try another size (Mission 2)
    python train.py --team 12 --no-norm           # Mission 2: what happens without normalisation
Writes model/model.pt and model/config.json, which app/main.py loads.

Recipe (each choice was tested with 5-fold cross-validation):
  1. drop glitch rows using the physical sensor limits in app/model.py
  2. hold out 20% of rows for validation; report baseline 1 (mean) and baseline 2 (linear)
  3. normalise the inputs with the TRAINING rows' mean and std
  4. learn log(days_left), not days_left: battery life is a product of factors, so errors are
     proportional (+-6%); in log space the target is smooth and the noise is the same size everywhere
  5. one hidden layer of 128 ReLUs, AdamW (weight decay 0.3) with a cosine learning-rate decay
  6. report the validation MAE, then retrain the same recipe on ALL clean rows and save that model
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, SENSOR_RANGES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="128", help="hidden widths, e.g. 128 or 32,32")
ap.add_argument("--epochs", type=int, default=800)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--wd", type=float, default=0.3, help="AdamW weight decay on the weights (0 = plain Adam)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-norm", action="store_true", help="skip input normalisation (Mission 2 comparison)")
ap.add_argument("--no-refit", action="store_true", help="save the 80%% model; don't retrain on all rows")
ap.add_argument("--out", default="model", help="folder to save model.pt and config.json in")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))
BATCH = 128

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

# ---- Mission 1: glitch rule -------------------------------------------------------------------
# A row is a glitch if any input is outside what the real sensor can report (SENSOR_RANGES,
# shared with the server) or if a value is missing. Each rule is safe because no working doorbell
# can produce such a reading: in our file they are 0.000 V batteries, -127 C temperatures (the
# DS18B20 "disconnected" code) and 0 dBm Wi-Fi (the "no reading" code). days_left must be > 0
# because a doorbell with a dead battery can't send a reading (and log(0) is undefined).
bad = df[FEATURES + ["days_left"]].isna().any(axis=1).to_numpy().copy()
for col, (lo, hi) in SENSOR_RANGES.items():
    out = ((df[col] < lo) | (df[col] > hi)).to_numpy()
    print(f"  glitch rule: {col:14s} outside [{lo:g}, {hi:g}]  -> {out.sum():3d} rows")
    bad = bad | out
bad = bad | (df.days_left <= 0).to_numpy()
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline 1 (predict the mean):  {mae(np.full_like(yv, y.mean()), yv):.2f} days")
# ---- Mission 1: baseline 2, a straight line through [X, 1] fitted by least squares -------------
A = np.c_[X, np.ones(len(X))].astype(np.float64)
w, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
lin_val = np.c_[Xv, np.ones(len(Xv))] @ w
print(f"baseline 2 (linear regression): {mae(lin_val, yv):.2f} days")


def scalers(X, y):
    """Mission 2: normalisation statistics, computed ONLY from the rows we train on.
    Using the validation rows here would leak information about them into training, and the
    server will see readings we never had, so it must use these same frozen numbers."""
    if a.no_norm:
        mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
    else:
        mu, sd = X.mean(0), X.std(0)
    t_mu, t_sd = float(np.log(y).mean()), float(np.log(y).std())   # the log target is standardised too
    return mu.astype(np.float32), sd.astype(np.float32), t_mu, t_sd


def to_target(y, t_mu, t_sd):
    """days_left -> what the network learns: standardised log(days)."""
    return ((np.log(y) - t_mu) / t_sd).astype(np.float32)


def from_output(o, t_mu, t_sd):
    """what the network outputs -> days_left (exp undoes the log, so it is never negative).
    app/main.py does exactly the same."""
    return np.exp(o * t_sd + t_mu)


def fit(Xn, t, log=None):
    """Train a fresh BatteryNet on normalised inputs Xn and targets t."""
    torch.manual_seed(a.seed)
    net = BatteryNet(hidden)
    # Weight decay pulls the weights (not the biases) towards zero every step: it keeps the
    # function smooth, so the network can't bend itself around the +-6% noise in the labels.
    weights = [p for n, p in net.named_parameters() if n.endswith("weight")]
    biases = [p for n, p in net.named_parameters() if n.endswith("bias")]
    opt = torch.optim.AdamW([{"params": weights, "weight_decay": a.wd},
                             {"params": biases, "weight_decay": 0.0}], lr=a.lr)
    # Cosine decay: big steps early to learn the shape, tiny steps at the end so the final
    # weights settle instead of bouncing around (the starter's val MAE jumped between epochs).
    steps = a.epochs * ((len(Xn) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    Xt, tt = torch.tensor(Xn), torch.tensor(t)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            loss = torch.nn.functional.mse_loss(net(Xt[b]), tt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (ep % 100 == 0 or ep == a.epochs - 1):
            net.eval(); log(ep, net)
    net.eval()
    return net


def predict(net, Xn, t_mu, t_sd):
    with torch.no_grad():
        return from_output(net(torch.tensor(Xn)).numpy(), t_mu, t_sd)


# ---- train on 80%, judge on the 20% the network never saw -------------------------------------
mu, sd, t_mu, t_sd = scalers(X, y)
Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
net = fit(Xn, to_target(y, t_mu, t_sd),
          log=lambda ep, n: print(f"  epoch {ep:4d}  val MAE {mae(predict(n, Xvn, t_mu, t_sd), yv):.2f} days"))
train_mae = mae(predict(net, Xn, t_mu, t_sd), y)
val_mae = mae(predict(net, Xvn, t_mu, t_sd), yv)
n_params = sum(p.numel() for p in net.parameters())
print(f"network {hidden} ({n_params} params): train MAE {train_mae:.2f}  val MAE {val_mae:.2f} days")

# ---- final model: same recipe, all clean rows -------------------------------------------------
# The validation run above has told us how good this recipe is. 25% more rows makes the model a
# little better on readings it has never seen, so the model we serve is retrained on everything.
# (Use --no-refit to save the 80% model instead.)
if not a.no_refit:
    X_all, y_all = df[FEATURES].to_numpy(np.float32), df.days_left.to_numpy(np.float32)
    mu, sd, t_mu, t_sd = scalers(X_all, y_all)
    Xa = (X_all - mu) / sd
    net = fit(Xa, to_target(y_all, t_mu, t_sd))
    print(f"refit on all {len(df)} clean rows: train MAE {mae(predict(net, Xa, t_mu, t_sd), y_all):.2f} days")

os.makedirs(a.out, exist_ok=True)
torch.save(net.state_dict(), f"{a.out}/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(), features=FEATURES,
               t_mu=t_mu, t_sd=t_sd,
               val_mae=round(val_mae, 2), train_mae=round(train_mae, 2),
               rows_used=int(len(tr) if a.no_refit else len(df)), glitch_rows=int(bad.sum())),
          open(f"{a.out}/config.json", "w"), indent=2)
print(f"saved {a.out}/model.pt and {a.out}/config.json")
