import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./output")
    args = parser.parse_args()

    results = []
    base_dir = Path(args.dir)

    for json_file in base_dir.rglob("results.json"):
        # Expected path pattern: output/DatasetName/Config_mlp/seed_X/shots_Y/results.json
        parts = json_file.parts
        
        try:
            # We assume a general structure ending in seed_X/shots_Y/results.json
            shot_part = [p for p in parts if p.startswith("shots_")][0]
            seed_part = [p for p in parts if p.startswith("seed_")][0]
            
            shots = int(shot_part.split("_")[1])
            seed = int(seed_part.split("_")[1])
            
            # Find index of the output directory to extract dataset and config
            out_idx = parts.index(base_dir.parts[-1]) if len(base_dir.parts) > 0 and base_dir.parts[-1] in parts else 0
            dataset = parts[out_idx + 1] if len(parts) > out_idx + 1 else "Unknown"
            config = parts[out_idx + 2] if len(parts) > out_idx + 2 else "Unknown"

            with open(json_file, 'r') as f:
                data = json.load(f)

            res = {
                "Dataset": dataset,
                "Config": config,
                "Shots": shots,
                "Seed": seed,
            }
            res.update(data)
            results.append(res)
            
        except Exception as e:
            print(f"Skipping {json_file}: {e}")

    if not results:
        print("No results found.")
        return

    # Use pandas to group and aggregate
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not installed. Install with `pip install pandas` for nice tables.")
        return
        
    df = pd.DataFrame(results)

    # We want mean and std for numeric columns.
    # Group by Dataset, Config, Shots
    metrics_cols = [c for c in df.columns if c not in ["Dataset", "Config", "Shots", "Seed"]]
    
    agg_funcs = {col: ["mean", "std", "count"] for col in metrics_cols}
    
    grouped = df.groupby(["Dataset", "Config", "Shots"], as_index=False).agg(agg_funcs)

    print("\n============ AGGREGATED RESULTS ============\n")
    # Pretty format logic
    for (dataset, config), group in grouped.groupby(["Dataset", "Config"]):
        print(f"[{dataset} | {config}]")
        
        # Build a simpler dataframe for display
        display_df = pd.DataFrame()
        display_df["Shots"] = group["Shots"]
        
        for col in metrics_cols:
            means = group[(col, "mean")]
            stds = group[(col, "std")]
            counts = group[(col, "count")]
            
            formatted = []
            for m, s, c in zip(means, stds, counts):
                if pd.isna(m):
                    formatted.append("-")
                else:
                    formatted.append(f"{m*100.:.2f} ± {s:.2f} (n={c})")
            display_df[col] = formatted
            
        print(display_df.to_string(index=False))
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()