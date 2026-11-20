# Model: GTA VI trophy rarity forecast

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast each of the
game's 55 PS5 trophies' Exophase earn rate at 2026-11-26T00:00Z as five
quantiles, scored by mean pinball loss on the logit scale.

## The problem in one paragraph

A trophy's rate is earners / players_tracked, and both move over launch
week. Earners grow as players progress (fast for story trophies, slow for
grinds); tracked players grow as more people sync their profiles (diluting
every rate). Exophase's page also refreshes in jumps, so the recorded reads
freeze for hours and then leap. The final case gives ~2.5 days of reads and
asks for rates 4.5 days ahead -- a long extrapolation from short, jumpy data.

## Median: a three-way blend (in logit space)

1. **Persistence** (weight 0.05): the rate stays where the last read put it.
2. **Trend** (weight 0.85): the rate's logit is linear in log(1 + hours
   since launch), fitted by recency-weighted least squares over the last
   24h of distinct reads. Adoption is roughly lognormal-shaped, so this is
   nearly the right functional form -- it is the strongest single view.
3. **Structure** (weight 0.10): log earners linear in log-time over the
   last 36h (slopes shrunk toward the tier mean, clamped to [0, 3]),
   divided by the forecasted tracked-player count. This is the view that
   sees dilution: when player growth bends, every rate bends with it.

Tracked players N(t): log N linear in log-time over the full history,
slope blended toward the last 24h (weight 0.8) and damped x0.6, because
adoption decelerates and a raw log-time slope over-predicts N (measured
+0.30 log bias before damping on the backtest).

Reads that repeat the previous read exactly (a frozen page between
refreshes) are deduplicated first. Fits use recent windows only, so the
launch-night reads -- a frozen page with a handful of accounts showing ~1%
for nearly every trophy -- cannot distort the trends. With a flat history
(the practice cases) the forecast is persistence with a wide,
refresh-aware spread.

## Spreads (logit units, combined in quadrature)

- random walk: 0.60 x sqrt(days of horizon);
- each fit's prediction-interval sigma (residual x leverage);
- refresh risk: 0.35 (the answering read can sit a whole refresh away);
- disagreement: 0.50 x |trend median - structural median|.

Total spread is clamped to [0.60, 2.50]. A plain persistence fallback runs
if the model itself ever throws, so a case is never left invalid.

## Backtest (honest scope)

The kit's practice cases have FLAT histories (7-8 identical reads), so they
validate output shape and determinism only -- not trend modelling. The model
was instead validated on 40 synthetic launch weeks: a convolution simulator
(player arrivals x lognormal time-to-earn per trophy, power-law adoption
with deceleration, Exophase-style refresh staleness), calibrated to the real
Exophase snapshot of 2026-11-20 10:37 UTC (Chapter 1 7.99%, most trophies
0.01-0.02%, Gone Fishing 100%). Hyperparameters were tuned on 20 worlds and
validated on 20 fresh worlds.

Mean pinball loss, main case (4.5-day horizon), fresh worlds:

| model | loss |
|---|---|
| this entry | **0.629** |
| house curve baseline (logit trend, last 24h) | 0.756 |
| house starter baseline (persistence) | 1.018 |

90% coverage of this entry on fresh worlds: 0.887 (nominal 0.90). The tuned
configuration also wins on the tuning worlds (0.639 vs 0.710 / 1.030), so
the edge is not grid overfitting.

Caveats: the backtest tests the METHOD under realistic dynamics, not the
true GTA numbers -- the real week could differ (a bigger launch-weekend
surge, a different refresh cadence). At 1-day horizons the starter baseline
still wins (0.341 vs 0.373): over a day, rates barely move and persistence
is hard to beat. The final's 4.5-day horizon is where the trend model earns
its keep.
