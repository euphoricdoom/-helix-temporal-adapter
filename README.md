# Helix Temporal Adapter

A tiny AC/DC temporal memory layer that helps sequence models remember what matters without retraining the whole system.

## The Problem

Machine learning systems often learn new tasks by overwriting old ones. This is catastrophic forgetting: the model improves on the latest task while silently damaging performance on earlier tasks.

Sequence tasks make the problem worse. If time is flattened into one static vector, the model loses the rhythm of events, markers, memory slots, and delayed decisions.

## The Solution

Helix adds a compact temporal adapter between reservoir features and task-specific readouts.

It uses two complementary streams:

- **AC stream:** fast event response for sudden changes and timestep-level signals.
- **DC stream:** slow accumulator memory for durable task state.
- **Learned gates:** adaptive write controls that decide what should enter memory.

The result is a small temporal memory layer that can preserve task isolation while adding sequence awareness.

## Results

From the current architecture notes:

- ✅ 100% accuracy on Copy task
- ✅ 100% accuracy on Parity task
- ✅ 100% accuracy on Adding task
- ✅ 0.0000 max forgetting across tested kernel counts

These results come from the extracted `project-512d` Helix benchmark path. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

## Quick Start

```python
import numpy as np
from helix.temporal import HelixTemporalAdapter

adapter = HelixTemporalAdapter(input_dim=512, projection_dim=256, seed=42)
for features_t in np.random.randn(10, 512):
    adapter.step(features_t)
sequence_features = adapter.final_features()
```

## Installation

This package is not on PyPI yet.

```bash
git clone https://github.com/euphoricdoom/-helix-temporal-adapter.git
cd -helix-temporal-adapter
pip install -e .
```

Install dependencies directly:

```bash
pip install -r requirements.txt
```

## Architecture

Helix projects high-dimensional feature streams into a compact temporal space, tracks fast AC state and slow DC state, then emits final trajectory features for downstream readouts.

Final features combine:

```text
last_projected + mean_projected + max_projected + ac_final + dc_final + ac*dc + phase
```

For deeper design notes, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Examples

See `examples/` directory for:

- Visual AC/DC stream demo — Coming Day 3
- Learned gate demo — Coming Day 4
- Benchmark-style adapter demo — Coming Day 5

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md) - Coming Day 6
- [Integration Guide](docs/INTEGRATION.md) - Coming Day 7

## Citation

If you use Helix Temporal Adapter in research or experiments, cite this repository:

```bibtex
@software{helix_temporal_adapter_2026,
  title = {Helix Temporal Adapter: AC/DC Dual-Stream Temporal Processing with Learned Gates},
  author = {Sowers, Carl},
  year = {2026},
  url = {https://github.com/euphoricdoom/-helix-temporal-adapter},
  license = {MIT}
}
```

## License

MIT
