# Tests Layout

The test tree is split by execution expectations.

- `smoke/`: the default `v0.1` green baseline
- `integration/`: opt-in distributed or GPU-heavy validation

Default local smoke command:

```bash
python -m pytest -q
```

Opt-in integration collection:

```bash
NANGILA_RUN_INTEGRATION=1 python -m pytest
```
