# coastal-dynamics 🌊

> **Coastal flood and mangrove migration models built on top of [DisSModel](https://github.com/LambdaGeo/dissmodel).**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![DisSModel](https://img.shields.io/badge/DisSModel-%3E%3D0.3.0-orange.svg)](https://github.com/LambdaGeo/dissmodel)
[![LambdaGeo](https://img.shields.io/badge/LambdaGeo-Research-green.svg)](https://github.com/LambdaGeo)

---

## 📖 About

**coastal-dynamics** implements spatially explicit models of coastal ecosystem processes using the **[DisSModel](https://github.com/LambdaGeo/dissmodel)** framework. Two coupled processes are modelled:

1. **Flood Dynamics** — sea-level rise propagation and terrain elevation adjustments.
2. **Mangrove Migration** — ecosystem response to rising sea levels, soil transitions, and sediment accretion.

The same scientific logic runs on **two spatial substrates**, illustrating DisSModel's environment-agnostic architecture:

| Substrate | Representation | Engine | Entry point |
|-----------|---------------|--------|-------------|
| **Raster** | GeoTIFF → `RasterBackend` | NumPy vectorized | `examples/main_raster.py` |
| **Vector** | Shapefile → `GeoDataFrame` | Iterative polygons | `examples/main_vector.py` |

---

## 🚀 Quick Start

### CLI local (development)

```bash
# Raster simulation (NumPy-based, fast)
python examples/main_raster.py run \
  --input  examples/data/input/synthetic_grid_60x60_tiff.zip \
  --format tiff \
  --output examples/data/output/saida.tiff \
  --param  interactive=true \
  --param  end_time=20

# Vector simulation (GeoDataFrame-based)
python examples/main_vector.py run \
  --input  examples/data/input/synthetic_grid_60x60_shp.zip \
  --output examples/data/output/saida.gpkg \
  --param  interactive=true \
  --param  end_time=20

# Load calibrated parameters from TOML
python examples/main_raster.py run \
  --input  examples/data/input/synthetic_grid_60x60_tiff.zip \
  --format tiff \
  --toml   examples/model.toml

# Validate executor data contract without running
python examples/main_raster.py validate \
  --input examples/data/input/synthetic_grid_60x60_tiff.zip

# Run Benchmark suite (Vector vs Raster mathematical equivalence)
python examples/main_benchmark.py run \
  --input  examples/data/input/synthetic_grid_60x60_shp.zip \
  --output ./benchmark/ \
  --param  end_time=10 \
  --param  tolerance=0.05

# Show resolved parameters
python examples/main_raster.py show --toml examples/model.toml
```

### Platform API (production / reproducibility)

```bash
# Submit job
curl -X POST http://localhost:8000/submit_job \
  -H "X-API-Key: chave-sergio" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name":    "coastal_raster",
    "input_dataset": "s3://dissmodel-inputs/mangue_grid.shp",
    "parameters":    {"end_time": 88, "taxa_elevacao": 0.011}
  }'

# Check status
curl -H "X-API-Key: chave-sergio" \
  http://localhost:8000/job/<experiment_id>

# Reproduce exact experiment
curl -X POST http://localhost:8000/experiments/<id>/reproduce \
  -H "X-API-Key: chave-sergio"
```

---

## 🧩 Model Processes

### 🌊 Flood Dynamics (`flood_model.py`)

Sea-level rise propagates across the landscape using a push-based neighbourhood algorithm.

- Sea level increases at a configurable rate (default: `0.011` m/year — IPCC RCP8.5).
- Flooded cells propagate water to lower-elevation neighbours.
- Terrain elevation adjusts dynamically due to water flux.

### 🌿 Mangrove Migration (`mangrove_model.py`)

Ecosystem transitions driven by tidal influence and flooding thresholds.

- Inland mangrove migration triggered when flooding reaches a tidal height threshold.
- Soil type transitions from mainland soil to mangrove mud.
- Optional sediment accretion based on Alongi (2008).

---

## 🗂️ Executor Architecture

coastal-dynamics follows the DissModel `ModelExecutor` pattern — each executor separates science from infrastructure. The same model runs locally via CLI or on the platform via API without changing a single line.

```
Science Layer (Model / Salabim)
  FloodModel, MangroveModel
  → only knows math, geometry and time

Infrastructure Layer (ModelExecutor)
  CoastalRasterExecutor, CoastalVectorExecutor, CoastalBenchmarkExecutor
  → only knows URIs, MinIO, band_map, parameters
```

### Executors available

| name | Substrate | Input → Output | Description |
|------|-----------|----------------|-------------|
| `coastal_raster` | RasterBackend / NumPy | Shapefile / GeoTIFF → GeoTIFF | Production raster simulation |
| `coastal_vector` | GeoDataFrame | Shapefile → GeoPackage | Production vector simulation |
| `coastal_validation` | Both | Shapefile → MD + PNG | Vector vs Raster mathematical equivalence |

### Benchmark / Validation executor

The `CoastalBenchmarkExecutor` (name: `coastal_validation`) runs vector and raster substrates simultaneously and compares results cell by cell. The architecture guarantees **100% match** across land use (`uso`), soil (`solo`) and elevation (`alt`) bands — validating that models are completely substrate-agnostic.

```bash
python examples/main_benchmark.py run \
  --input  examples/data/input/synthetic_grid_60x60_shp.zip \
  --output ./benchmark/ \
  --param  end_time=10 \
  --param  tolerance=0.05
```

Output:
```
benchmark/
  report.md    ← runtime comparison + accuracy metrics (match %, MAE, RMSE) per band
  scatter.png  ← Vector vs Raster scatter plots for uso, solo, alt
```

### model.toml — simulation parameters

```toml
# examples/model.toml

[model.parameters]
end_time      = 88
taxa_elevacao = 0.011     # m/year — IPCC RCP8.5
altura_mare   = 6.0       # m
acrecao_ativa = false
resolution    = 100.0     # m — raster only
crs           = "EPSG:31984"
interactive   = false
```

---

## 📦 Installation

```bash
# From source
git clone https://github.com/lambdageo/coastal-dynamics.git
cd coastal-dynamics
pip install -e .

# From GitHub branch (platform / dissmodel-configs)
pip install "git+https://github.com/LambdaGeo/coastal-dynamics.git@develop"
```

**Dependencies:** `dissmodel >= 0.3.0`, `numpy`, `geopandas`, `rasterio`, `shapely`, `matplotlib`

---

## 🗂️ Project Structure

```
coastal-dynamics/
├── coastal_dynamics/
│   ├── __init__.py
│   ├── executor/                         # ModelExecutor implementations
│   │   ├── __init__.py                   # imports executors → auto-registration
│   │   ├── coastal_raster_executor.py    # RasterBackend/NumPy substrate
│   │   ├── coastal_vector_executor.py    # GeoDataFrame substrate
│   │   └── coastal_validation_executor.py  # Vector vs Raster comparison
│   ├── raster/                           # NumPy-based models
│   │   ├── flood_model.py
│   │   └── mangrove_model.py
│   ├── vector/                           # GeoDataFrame-based models
│   │   ├── flood_model.py
│   │   └── mangrove_model.py
│   └── common/
│       └── constants.py                  # TIFF_BANDS, CRS, USO_COLORS, ...
├── examples/
│   ├── main_raster.py                    # CoastalRasterExecutor via CLI
│   ├── main_vector.py                    # CoastalVectorExecutor via CLI
│   ├── main_benchmark.py                 # CoastalBenchmarkExecutor via CLI
│   ├── model.toml                        # Simulation parameters
│   └── data/
│       ├── input/
│       │   ├── synthetic_grid_60x60_shp.zip
│       │   ├── synthetic_grid_60x60_tiff.zip
│       │   └── elevacao_pol.zip
│       └── output/
├── pyproject.toml
└── requirements.txt
```

---

## 🎯 Design Philosophy

1. **Dual substrate** — same scientific logic runs on both vector (GeoDataFrame) and raster (RasterBackend/NumPy).
2. **Executor pattern** — science layer never knows about files, URIs or cloud; infrastructure layer never calculates spatial equations.
3. **Benchmark-first** — `coastal_validation` validates substrate equivalence before any production use.
4. **Reproducibility** — each experiment records model commit, input checksum, and resolved spec via `ExperimentRecord`.
5. **Transparency** — simulation parameters live in `model.toml`, version-controlled separately from code.

---

## 🔬 Citation

If you use this project or the DisSModel architecture in your research, please cite:

```
Bezerra, R. (2014). Modelagem da migração de manguezais sob efeito da
elevação do nível do mar. INPE.

Costa, S. S. et al. (2026). DisSModel — A Python framework for spatial
discrete simulation models. LambdaGEO, UFMA.
```

---

## 🤝 Contributing

1. Fork the repository and create a feature branch
2. Implement changes and add tests
3. Submit a Pull Request with a clear description

To register a new model in the platform, open a PR in [dissmodel-configs](https://github.com/LambdaGeo/dissmodel-configs) with a TOML spec pointing to your package.

---

Developed by the **[LambdaGeo](https://lambdageo.github.io)** research group.


## include after

python coastal_dynamics/executor/coastal_validation_full_executor.py run   --input  examples/data/input/elevacao_pol.zip   --output examples/data/output/validation   --param  golden_dir=tests/fixtures/golden   --param  end_time=30   --param  taxa_elevacao=0.05   --param  altura_mare=6.0   --param  checkpoints=[1,5,10,15,20,25,30]