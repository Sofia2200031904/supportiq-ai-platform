import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    audit = json.loads((reports / "split_audit" / "audit.json").read_text())
    majority = json.loads((reports / "split_audit" / "majority_baseline.json").read_text())
    analytics = json.loads((reports / "analytics_validation_2025.json").read_text())
    preview = root / "preview"
    preview.mkdir(exist_ok=True)
    data = {"audit": audit, "majority": majority,
            "analytics": {key: analytics[key] for key in ("rows", "with_narrative", "duplicate_text", "months")}}
    (preview / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name in ("monthly", "products", "companies"):
        shutil.copyfile(reports / "eda" / f"{name}.png", preview / f"{name}.png")
    shutil.copyfile(reports / "split_audit" / "class_counts_after.csv", preview / "class_counts.csv")
    shutil.copyfile(reports / "eda" / "business_insights.md", preview / "business_insights.md")


if __name__ == "__main__":
    main()
