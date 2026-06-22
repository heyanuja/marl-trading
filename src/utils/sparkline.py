from __future__ import annotations

from collections.abc import Iterable

_BARS = "▁▂▃▄▅▆▇█"


def sparkline(xs: Iterable[float], *, width: int = 32) -> str:
    vals = [float(x) for x in xs]
    if not vals:
        return ""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    if len(vals) > width:
        step = len(vals) / width
        sampled: list[float] = []
        for i in range(width):
            sampled.append(vals[int(i * step)])
        vals = sampled

    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return _BARS[0] * len(vals)

    out = []
    for v in vals:
        t = (v - lo) / (hi - lo)
        idx = int(round(t * (len(_BARS) - 1)))
        idx = max(0, min(len(_BARS) - 1, idx))
        out.append(_BARS[idx])
    return "".join(out)

