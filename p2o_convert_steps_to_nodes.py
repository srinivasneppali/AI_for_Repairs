
import sys, yaml, io

def convert(in_path, out_path=None):
    with io.open(in_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "nodes" not in data and "steps" in data:
        nodes = {step["id"]: step for step in data["steps"]}
        data["nodes"] = nodes
        # keep start if present, else first step id
        if "start" not in data and data.get("steps"):
            data["start"] = data["steps"][0]["id"]
    # Optionally drop "steps" to avoid confusion
    if "steps" in data:
        del data["steps"]
    out_path = out_path or in_path.replace(".yaml", "_nodes.yaml")
    with io.open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print("Wrote:", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python p2o_convert_steps_to_nodes.py input.yaml [output.yaml]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
