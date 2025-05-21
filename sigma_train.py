import os
import json
import yaml
from uuid import uuid4
from datetime import date

SIGMA_RULE_DIR = "./data/sigma_all_rules/rules"
OUTPUT_FILE = "sigma_train_from_files.json"
examples = []

def rule_to_sample(rule, raw_text):
    title = rule.get("title", "Unknown rule")
    desc = rule.get("description", "")
    instruction = f"Generate a Sigma rule: {title}. {desc}"
    
    return {
        "instruction": instruction.strip(),
        "input": "",
        "output": raw_text.strip()
    }

# Collect all .yml files
for root, _, files in os.walk(SIGMA_RULE_DIR):
    for file in files:
        if file.endswith(".yml") or file.endswith(".yaml"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()
                    rule = yaml.safe_load(raw)
                    if isinstance(rule, dict) and "logsource" in rule and "detection" in rule:
                        sample = rule_to_sample(rule, raw)
                        examples.append(sample)
            except Exception as e:
                print(f"❌ Failed to parse {filepath}: {e}")

# Save JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(examples, out, indent=2)

print(f"✅ Extracted {len(examples)} training samples -> {OUTPUT_FILE}")
