import json
from pathlib import Path

def patch_reality_check():
    path = Path("analytics/ml/reality_check.py")
    content = path.read_text()
    if "json.dump(tests" not in content:
        content = content.replace("self._generate_master_report(tests, base_port, oof_df)", "self._generate_master_report(tests, base_port, oof_df)\n        with open(self.out_dir / 'tests_dump.json', 'w') as f:\n            import json; json.dump(tests, f, default=str)")
        path.write_text(content)

if __name__ == "__main__":
    patch_reality_check()
