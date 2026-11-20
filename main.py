"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's PS5 trophies, its Exophase earn rate at the
case's target as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by
mean pinball loss on the logit scale.

Model: the median (in logit space) blends three views of each trophy's rate:

  * persistence: the rate stays where the last read put it;
  * trend: the rate's own log-time trend over the last 24h of reads
    (adoption is roughly lognormal-shaped, so the logit rate is nearly
    linear in log-time);
  * structure: earners and tracked players modelled separately -- log
    earners linear in log-time over the last 36h, slopes shrunk toward
    their tier's mean -- with the rate their ratio at the target. This
    is the view that sees dilution: new players keep joining Exophase's
    tracked set, so a slow trophy's rate can fall even as its earners
    rise, and a bend in player growth bends every rate.

Tracked players N(t) are fitted the same way (log N linear in log-time,
slope blended toward the last 24h because adoption decelerates).

Spreads combine a random-walk term growing with sqrt(days), both fits'
prediction-interval uncertainty, a refresh term (Exophase's page updates
in jumps, so the answering read can sit a whole refresh from the trend),
and a term for how far the trend and structural medians disagree.

Reads that repeat the previous read exactly (a frozen page between refreshes)
are deduplicated first, so a long frozen stretch does not overweight one
point. With a flat or single-read history no trend is identifiable: the
forecast falls back to persistence with a wide, refresh-aware spread.

Standard library only. Deterministic: no randomness, no wall clock, no dict
iteration order dependence; identical input bytes give identical output bytes.
Any unexpected failure inside the model falls back to a plain persistence
forecast, so a case is never left invalid by a modelling bug.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- Hyperparameters (tuned on a synthetic adoption backtest: 20 tuning
# worlds + 20 fresh validation worlds; the chosen set wins on both) ---
SIGMA0 = 0.60          # logit spread per sqrt-day of horizon
SIGMA_REFRESH = 0.35   # extra logit spread for Exophase refresh jumps
SIGMA_FLOOR = 0.60     # minimum total logit spread
SIGMA_CAP = 2.50       # maximum total logit spread
SIGMA_FLAT = 1.20      # spread when the history is flat (no trend visible)
HALF_LIFE_H = 18.0     # recency half-life (hours) for the fits
SHRINK_K = 12.0        # shrinkage strength for per-trophy slopes
SLOPE_CAP = 3.0        # max log-time slope for earners
N_GROWTH_CAP = 6.0     # max N(target) / N(last)
E_GROWTH_CAP = 40.0    # max E(target) / E(last)
RECENT_H = 24.0        # window for the recent-slope blend of N
N_RECENT_W = 0.80      # weight of the recent slope in the N blend
N_DAMP = 0.60          # damping on the N slope (adoption decelerates)
TREND_H = 24.0         # window for the rate-trend fit (hours)
STRUCT_H = 36.0        # window for the earners fit (hours)
W_TREND = 0.85         # blend weight: rate-trend median
W_STRUCT = 0.10        # blend weight: earners/tracked median
# (persistence takes the remaining weight)
DISAGREE_K = 0.50      # spread added per logit of model disagreement
TREND_MAX_MOVE = 3.0   # max logit move the trend may predict
RATE_LO = 0.01         # output floor (logit is undefined at 0)
RATE_HI = 99.99        # output cap (logit is undefined at 100)


# --- Small numeric helpers ---

def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                                + b[4]) * r + 1)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def z_scores(levels):
    if levels == LEVELS:
        return list(Z)
    return [norm_ppf(p) for p in levels]


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparsable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def log_time(dt):
    """log(1 + hours since launch); the time axis every fit uses."""
    hours = (dt - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (intercept, slope, resid_std, xbar_w, sxx_w) or None when the fit
    is degenerate (fewer than 2 points or no spread in x).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    dof = max(n - 2, 1)
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / dof)
    return (intercept, slope, resid, xbar, sxx)


def pred_sigma(fit, x_target, n_points):
    """Prediction-interval sigma at x_target for a wls fit."""
    if fit is None:
        return 1.0
    _, _, resid, xbar, sxx = fit
    leverage = 1.0 / max(n_points, 1) + (x_target - xbar) ** 2 / sxx
    return resid * math.sqrt(1.0 + leverage)


# --- Case handling ---

def distinct_reads(history):
    """Collapse consecutive identical reads (frozen page between refreshes).

    Returns a list of (time, players_tracked, rates) keeping the FIRST read
    of each distinct state, in order. Reads without a usable time or player
    count are skipped for fitting (the last read is still found separately).
    """
    out = []
    prev_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        players = read.get("players_tracked")
        rates = read.get("rates")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        when = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if when is None:
            continue
        out.append((when, float(players), rates))
    return out


def last_rates(history, n_trophies):
    """The last read's rates (None-padded), whatever its time was."""
    for read in reversed(history):
        if isinstance(read, dict) and isinstance(read.get("rates"), list):
            rates = read["rates"]
            return [rates[i] if i < len(rates) else None
                    for i in range(n_trophies)]
    return [None] * n_trophies


def window_points(reads, t_last, hours):
    """Indices of reads within `hours` before t_last (at least the last)."""
    idx = [i for i, (when, _, _) in enumerate(reads)
           if (t_last - when).total_seconds() <= hours * 3600.0]
    return idx if idx else [len(reads) - 1]


def fit_log_series(times, values, t_last, half_life):
    """Recency-weighted linear fit of values against log-time.

    Returns (fit, x_last) with fit as wls() gives it, or (None, None).
    """
    if len(times) < 2:
        return None, None
    xs = [log_time(t) for t in times]
    ws = [0.5 ** ((t_last - t).total_seconds() / 3600.0 / half_life)
          for t in times]
    return wls(xs, values, ws), xs[-1]


def forecast_case(case):
    """Forecast one case. Returns {"trophies": [{"id", "q": [...]}]}.

    Median (in logit space) blends three views:
      * persistence: the last rate (no movement);
      * trend: the rate's own log-time trend over the last 24h of reads;
      * structure: earners and tracked players modelled separately, the
        rate being their ratio at the target.
    The blend hedges model misspecification: the trend is well specified
    when adoption is lognormal-shaped, the structural view wins when the
    tracked-player growth bends (dilution changes pace). Spreads combine
    horizon growth, both fits' prediction uncertainty, refresh risk, and
    a term for how far the two models disagree.
    """
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    levels = case.get("levels") or LEVELS
    zs = z_scores(levels)
    target_dt = parse_time(case.get("target"))
    x_target = log_time(target_dt) if target_dt is not None else None

    last = last_rates(history, len(trophies))
    reads = distinct_reads(history)
    flat = len(reads) < 2 or x_target is None
    base_sigma = SIGMA0 * math.sqrt(horizon_days)

    # --- Tracked players N(t): full-history fit, slope blended recent ---
    # Log N is concave in log-time (adoption decelerates), so a straight
    # line over-estimates future growth. Two guards: the slope is blended
    # toward the last-24h slope, and the fitted slope is damped by N_DAMP.
    n_target = None
    n_sigma = 1.0
    if reads:
        n_last = reads[-1][1]
        n_target = n_last
        if len(reads) >= 2 and x_target is not None:
            t_last = reads[-1][0]
            fit, x_last = fit_log_series(
                [w for w, _, _ in reads],
                [math.log(n) for _, n, _ in reads], t_last, HALF_LIFE_H)
            if fit is not None:
                n_sigma = pred_sigma(fit, x_target, len(reads))
                slope = fit[1]
                ridx = window_points(reads, t_last, RECENT_H)
                if len(ridx) >= 3:
                    rfit, _ = fit_log_series(
                        [reads[i][0] for i in ridx],
                        [math.log(reads[i][1]) for i in ridx],
                        t_last, HALF_LIFE_H)
                    if rfit is not None:
                        slope = ((1.0 - N_RECENT_W) * fit[1]
                                 + N_RECENT_W * rfit[1])
                slope *= N_DAMP
                log_n = math.log(n_last) + slope * (x_target - x_last)
                n_target = math.exp(min(
                    log_n, math.log(n_last * N_GROWTH_CAP)))
                n_target = max(n_target, n_last)

    # --- Per-trophy trend + structural fits (recent windows only) ---
    # The early launch-night reads are a different regime (a frozen page
    # with a handful of accounts); recent windows keep them out of the fits.
    trend = []   # (median_logit, sigma) or None
    struct = []  # (median_logit, sigma) or None
    tier_slopes = {}
    raw_struct = []
    if not flat:
        t_last = reads[-1][0]
        for i, trophy in enumerate(trophies):
            # Trend: logit(rate) vs log-time, last TREND_H hours.
            tidx = window_points(reads, t_last, TREND_H)
            if len(tidx) < 3:
                tidx = window_points(reads, t_last, TREND_H * 2)
            tpts = [(reads[j][0], reads[j][2][i]) for j in tidx
                    if i < len(reads[j][2]) and reads[j][2][i] is not None]
            if len(tpts) >= 2:
                tfit, tx_last = fit_log_series(
                    [w for w, _ in tpts],
                    [logit(r) for _, r in tpts], t_last, HALF_LIFE_H)
            else:
                tfit, tx_last = None, None
            trend.append((tfit, tx_last, tpts))
            # Structure: log(E + 0.5) vs log-time, last STRUCT_H hours.
            sidx = window_points(reads, t_last, STRUCT_H)
            if len(sidx) < 3:
                sidx = window_points(reads, t_last, STRUCT_H * 2)
            spts = [(reads[j][0], reads[j][2][i] / 100.0 * reads[j][1])
                    for j in sidx
                    if i < len(reads[j][2]) and reads[j][2][i] is not None]
            if len(spts) >= 2:
                sfit, sx_last = fit_log_series(
                    [w for w, _ in spts],
                    [math.log(e + 0.5) for _, e in spts],
                    t_last, HALF_LIFE_H)
            else:
                sfit, sx_last = None, None
            raw_struct.append((sfit, sx_last, spts))
            if sfit is not None:
                tier_slopes.setdefault(trophy.get("tier"), []).append(sfit[1])
        tier_mean = {t: sum(v) / len(v)
                     for t, v in tier_slopes.items() if v}
        all_slopes = [s for v in tier_slopes.values() for s in v]
        global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0

    w_trend = W_TREND
    w_struct = W_STRUCT
    w_persist = max(0.0, 1.0 - w_trend - w_struct)
    total_w = w_trend + w_struct + w_persist
    w_trend, w_struct, w_persist = (w / total_w for w in
                                    (w_trend, w_struct, w_persist))

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last[i]
        if rate_last is None:
            rate_last = 0.0
        m_persist = logit(rate_last if rate_last > 0 else RATE_LO)
        if flat:
            median = clamp_rate(rate_last if rate_last > 0 else RATE_LO)
            sigma = max(base_sigma, SIGMA_FLAT)
        else:
            # Trend median.
            tfit, tx_last, tpts = trend[i]
            if tfit is not None:
                move = tfit[1] * (x_target - tx_last)
                move = min(max(move, -TREND_MAX_MOVE), TREND_MAX_MOVE)
                m_trend = m_persist + move
                s_trend = pred_sigma(tfit, x_target, len(tpts))
            else:
                m_trend, s_trend = m_persist, 1.0
            # Structural median.
            sfit, sx_last, spts = raw_struct[i]
            if sfit is not None and n_target:
                n_pts = len(spts)
                w_self = n_pts / (n_pts + SHRINK_K)
                prior = tier_mean.get(trophy.get("tier"), global_mean)
                slope = w_self * sfit[1] + (1.0 - w_self) * prior
                slope = min(max(slope, 0.0), SLOPE_CAP)
                e_last = spts[-1][1]
                log_e = (math.log(e_last + 0.5)
                         + slope * (x_target - sx_last))
                e_target = math.exp(log_e) - 0.5
                e_target = min(max(e_target, 0.0),
                               max(e_last, 0.5) * E_GROWTH_CAP)
                m_struct = logit(100.0 * e_target / n_target)
                s_struct = math.sqrt(pred_sigma(sfit, x_target, n_pts) ** 2
                                     + n_sigma ** 2)
            else:
                m_struct, s_struct = m_persist, 1.0
            centre = (w_persist * m_persist + w_trend * m_trend
                      + w_struct * m_struct)
            median = clamp_rate(inv_logit(centre))
            disagree = abs(m_trend - m_struct)
            sigma = math.sqrt(base_sigma ** 2
                              + (w_trend * s_trend) ** 2
                              + (w_struct * s_struct) ** 2
                              + SIGMA_REFRESH ** 2
                              + (DISAGREE_K * disagree) ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        centre = logit(median)
        qs = []
        prev = None
        for z in zs:
            q = round4(inv_logit(centre + z * sigma))
            q = min(max(q, 0.0), 100.0)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Plain persistence forecast: last rate, random-walk spread.

    Used only when the model itself fails, so a case is never invalid.
    """
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    levels = case.get("levels") or LEVELS
    zs = z_scores(levels)
    last = last_rates(history, len(trophies))
    spread = max(0.5 * math.sqrt(horizon_days), 0.6)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last[i]
        median = clamp_rate(rate if rate else RATE_LO)
        centre = logit(median)
        qs = []
        prev = None
        for z in zs:
            q = round4(inv_logit(centre + z * spread))
            q = min(max(q, 0.0), 100.0)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
