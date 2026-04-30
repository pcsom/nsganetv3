# Run + Visualization Guide (Supernet-Free Oxford Flowers)

This guide is the quick operational path for producing class-ready NAS results and Pareto visualization.

## 1) Submit a meaningful run on SLURM

Use a config with more than smoke-test settings (example: 8-12 iterations, larger pop/gens than smoke).

```bash
ts=$(date +%Y%m%d_%H%M%S)
save_dir="/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv3/class_significant_${ts}"

sbatch --time=04:00:00 \
  run_nsganetv3_slurm.sh \
  -s "$save_dir" \
  -c ".tmp/class_significant_config.toml" \
  -f "/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv2/prod_500/prod_500_results.csv" \
  -m "virtual" \
  -e "nsganetv2-llm"
```

## 2) Monitor live progress

```bash
squeue -j <job_id> -o "%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R"
log_file="/storage/ice-shared/vip-vvk/data/AOT/$USER/evolution_logs/nsganet.<job_id>.4294967294.log"
tail -f "$log_file"
```

Healthy signs:

- `NSGANetV3 search (virtual mode) on flowers102`
- `iter X/Y`
- `predictor progress: ...`
- `iter_X.stats` writes
- final `NSGANetV3 search completed`

## 3) Generate Pareto visualization (with Pareto line)

After completion, use final iteration stats:

```bash
conda run -n nsganetv2-llm python visualize_pareto.py \
  --stats "$save_dir/iter_<final_iter>.stats" \
  --output "$save_dir/pareto_front_iter<final_iter>.png" \
  --pareto_json "$save_dir/pareto_front_iter<final_iter>.json"
```

What this now does:

- computes non-dominated front on objective matrix `(complexity, top1_error)`
- plots all points as gray cloud
- plots Pareto points in red
- draws a black line through Pareto points sorted by complexity
- optionally writes sorted Pareto coordinates to json

### Regenerate from latest completed run

If you already have a completed run and only want to regenerate visualization:

```bash
save_dir="/storage/ice-shared/vip-vvk/data/AOT/$USER/nsganetv3/class_significant_20260429_211745"
final_iter=10

conda run -n nsganetv2-llm python visualize_pareto.py \
  --stats "$save_dir/iter_${final_iter}.stats" \
  --output "$save_dir/pareto_front_iter${final_iter}_with_line.png" \
  --pareto_json "$save_dir/pareto_front_iter${final_iter}_with_line.json"
```

Verify files:

```bash
ls -lh \
  "$save_dir/pareto_front_iter${final_iter}_with_line.png" \
  "$save_dir/pareto_front_iter${final_iter}_with_line.json"
```

## 4) Quick result check

```bash
conda run -n nsganetv2-llm python - <<'PY'
import json
p = "$save_dir/iter_<final_iter>.stats"
d = json.load(open(p, "r", encoding="utf-8"))
print("archive_len", len(d.get("archive", [])))
print("candidates_len", len(d.get("candidates", [])))
print("hv", d.get("hv"))
print("mode", d.get("evaluation_mode"))
print("surrogate", d.get("surrogate", {}).get("name"))
PY
```
