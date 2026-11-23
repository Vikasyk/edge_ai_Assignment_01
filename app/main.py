"""
app/main.py -- the battery API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 12, "params": 897, "val_mae": ...}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, SENSOR_RANGES, BatteryNet, layer_table

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
# Inference runs in float64. The answer for one reading is then the same whether it comes alone
# (/predict) or inside a batch (/predict_batch): float32 sums done in a different order can differ
# in the last digit, and that could flip the 2-decimal rounding. The stored weights are float32.
net.double()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float64), np.array(cfg["sd"], np.float64)
T_MU, T_SD = cfg["t_mu"], cfg["t_sd"]  # the network predicts standardised log(days_left)
N_PARAMS = sum(p.numel() for p in net.parameters())
MAX_DAYS = 2000.0  # safety cap for wild extrapolation; real answers are far below it


def in_range(name):
    """Mission 3: a field that only accepts what the real sensor can report (see app/model.py)."""
    lo, hi = SENSOR_RANGES[name]
    return Field(ge=lo, le=hi, allow_inf_nan=False)


class Reading(BaseModel):
    # Glitches like 0.0 V, -127 C or 0 dBm are refused with HTTP 422 instead of being answered.
    battery_v: float = in_range("battery_v")          # 2.5 .. 4.5 V
    temp_c: float = in_range("temp_c")                # -40 .. 80 C
    rings_per_day: float = in_range("rings_per_day")  # 0 .. 200
    wifi_rssi_dbm: float = in_range("wifi_rssi_dbm")  # -100 .. -10 dBm
    age_months: float = in_range("age_months")        # 0 .. 120 months


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API (team 12, Parallax)")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)

# We read the raw body ourselves (so "parse" can be timed), which hides the body from /docs.
# These two descriptions put it back, so /docs shows a ready-to-edit reading under "Try it out".
EXAMPLE = {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}
READING_SCHEMA = Reading.model_json_schema()
DOCS_ONE = {"requestBody": {"required": True, "content": {"application/json": {
    "schema": READING_SCHEMA, "example": EXAMPLE}}}}
DOCS_BATCH = {"requestBody": {"required": True, "content": {"application/json": {
    "schema": {"type": "object", "required": ["rows"],
               "properties": {"rows": {"type": "array", "items": READING_SCHEMA}}},
    "example": {"rows": [EXAMPLE, dict(EXAMPLE, battery_v=3.52, rings_per_day=12)]}}}}}


def to_days(out):
    """Network output -> days left: undo the standardisation, then exp() undoes the log.
    Must match from_output() in train.py."""
    return np.clip(np.exp(out * T_SD + T_MU), 0.0, MAX_DAYS)


# Warm-up: the very first forward pass in a fresh process is slow (memory allocation, kernel
# selection). Do it once at start-up so the first real request doesn't pay for it.
with torch.no_grad():
    net(torch.zeros(1, len(FEATURES), dtype=torch.float64))


@app.get("/model_info")
def model_info():
    # Mission 2 + 4: the per-layer table, worked out from net.net (see layer_table in app/model.py),
    # so it stays right if the layer sizes change.
    return layer_table(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        # The reason (which field, which limit) without echoing the input: an input like NaN
        # can't be written back out as JSON and would turn a clean 422 into a 500.
        detail = json.loads(e.json(include_input=False, include_url=False))
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": detail})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]

    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float64) - MU) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in to_days(y)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict", openapi_extra=DOCS_ONE)
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch", openapi_extra=DOCS_BATCH)
async def predict_batch(request: Request):
    return await run(request, Batch)
