import ast
import csv


def _parse_sequence(text_value):
    if text_value is None:
        return []
    if isinstance(text_value, list):
        return text_value
    value = str(text_value).strip()
    if not value:
        return []
    return ast.literal_eval(value)


def _first_present_float(row, keys):
    for key in keys:
        raw = row.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        return float(value)
    return None


def _proxy_complexity(arch, sec_obj):
    proxy =float(sum(arch["ks"]) * sum(arch["e"]) * sum(arch["d"]) * (int(arch["r"]) ** 2))
    flops = proxy / 1e7
    params =proxy/ 5e8
    if sec_obj == "params":
        return params
    return flops

def load_offline_ground_truth(csv_path, sec_obj, n_classes, supernet_path):
    _ = supernet_path
    _ = n_classes
    print(f"Loading offline ground truth from {csv_path}...")
    archive = []
    fallback =0
    processed = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            processed+= 1
            status = str(row.get("status", "success")).strip().lower()
            if status and status != "success":
                continue
            ks = _parse_sequence(row.get("ks"))
            e = _parse_sequence(row.get("e"))
            d = _parse_sequence(row.get("d"))
            r_raw = row.get("r")
            if r_raw is None or str(r_raw).strip() == "":
                continue
            arch_dict = {"ks": ks, "e": e, "d": d, "r": int(float(r_raw))}
            top1 = _first_present_float(row, ("best_top1", "final_top1", "top1"))
            if top1 is None:
                continue
            complexity = _first_present_float(
                row,
                (sec_obj, f"best_{sec_obj}", "complexity", "flops", "params", "cpu", "gpu"),
            )
            if complexity is None:
                complexity= float(_proxy_complexity(arch_dict, sec_obj))
                fallback+= 1
            archive.append((arch_dict, 100.0 - top1, complexity))
            if processed % 100 == 0:
                print(f"  parsed {processed} rows, kept {len(archive)}", flush=True)
    print(f"Successfully loaded {len(archive)} architectures from offline data.")
    if fallback:
        print(f"  Computed {sec_obj} via static profiler for {fallback} rows (CSV lacked column).")
    return archive
