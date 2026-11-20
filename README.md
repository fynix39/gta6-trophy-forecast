# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fynix39/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fynix39/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate one week
after launch, as five quantiles per trophy (0.05 / 0.25 / 0.5 / 0.75 / 0.95),
scored by mean pinball loss on the logit scale.

The median blends three views of each rate: persistence, the rate's own
log-time trend over the last 24 hours of reads, and a structural view that
models earners and tracked players separately (a slow trophy's rate can fall
even as its earners rise, because new players keep joining Exophase's
tracked set). Spreads combine horizon growth, both fits' uncertainty,
Exophase refresh risk, and how far the two models disagree. See MODEL.md
for the full design and the backtest results.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only. It reads every `*.in` case
file in the input directory and writes one `<id>.out` forecast per case.
Two runs on the same input produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jvcw07av&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
