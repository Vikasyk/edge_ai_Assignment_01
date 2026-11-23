"""model.py -- the network, shared by train.py and the server so both build the same thing.

Also holds the two things train.py and app/main.py must agree on:
  SENSOR_RANGES  what a real doorbell sensor can report (used to drop glitches AND to refuse them)
  layer_table()  the per-layer params / bytes / MACs table, computed from the network itself
"""
import torch
from torch import nn

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]

# Physical limits of a real reading (Mission 1 glitch rule + Mission 3 HTTP 422 ranges).
# These come from what the hardware can report, NOT from our data's min/max, so a real
# reading just outside our data (say 3.30 V or 48 C) still gets an answer.
SENSOR_RANGES = {
    # Li-ion cell: the protection circuit cuts out near 3.0 V and a full charge is 4.2 V.
    # 0.0 V means the ADC was never read: a doorbell at 0 V could not send anything.
    "battery_v": (2.5, 4.5),
    # Outdoor electronics are rated -40..+85 C. -127 C is the DS18B20 "sensor disconnected" code
    # and 85.0 C is its power-on reset value, so we stop just below 85.
    "temp_c": (-40.0, 80.0),
    # A count can't be negative; 200 rings a day is one every 7 minutes, all day and night.
    "rings_per_day": (0, 200),
    # Wi-Fi RSSI: below about -100 dBm there is no link; -10 dBm is touching the router.
    # 0 dBm is what the radio reports when it has no reading.
    "wifi_rssi_dbm": (-100, -10),
    # A doorbell can't be younger than new; 10 years is past any battery doorbell's life.
    "age_months": (0, 120),
}


class BatteryNet(nn.Module):
    def __init__(self, hidden=(128,)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


SRAM_BUDGET_BYTES = 300 * 1024  # ESP32-S3: ~512 KB SRAM, roughly 300 KB free for our program


def layer_table(net):
    """Per-layer cost of one prediction, read from the network's own Linear layers.

    Linear(a, b): a*b weights + b biases = params; a*b multiply-accumulates (MACs);
    float32 = 4 bytes per param, int8 = 1 byte per param. ReLU has no parameters.
    """
    layers = []
    for m in net.net:
        if isinstance(m, nn.Linear):
            a, b = m.in_features, m.out_features
            params = a * b + b
            layers.append({"name": f"linear{len(layers) + 1}", "in": a, "out": b,
                           "params": params, "bytes_f32": 4 * params, "macs": a * b})
    total_params = sum(l["params"] for l in layers)
    total_f32 = sum(l["bytes_f32"] for l in layers)
    total_int8 = total_params
    return {"layers": layers,
            "total_params": total_params,
            "total_bytes_f32": total_f32,
            "total_bytes_int8": total_int8,
            "total_macs": sum(l["macs"] for l in layers),
            "fits_300kb_sram_f32": total_f32 <= SRAM_BUDGET_BYTES,
            "fits_300kb_sram_int8": total_int8 <= SRAM_BUDGET_BYTES}
