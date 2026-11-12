# Candidate-void finder

`find_voronoi_voids.py` generates periodic candidate-void sites (nearest-atom clearance maxima and empty-sphere centers) from a crystal structure. It is imported by `elf_void_analysis/build_graph_records.py` and the connectivity tools. The earlier exploratory benchmark and synthetic validation were removed; see git history before this cleanup.

```bash
pip install -r requirements.txt
python find_voronoi_voids.py --help
```
