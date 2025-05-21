from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import sys
from PyPDF2 import PdfReader

# === CONFIG ===
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_DIR = "mistral-sigma-lora"

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, LORA_DIR)

# === EXTRACT TEXT FROM PDF ===
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# === GENERATE SIGMA RULES ===
def generate_sigma_rules(text):
    prompt = f"""You are a cybersecurity analyst. Given the following threat report, generate as many Sigma rules as appropriate to detect the described behaviors.

Output ONLY valid Sigma rules in YAML format. Each rule must start with `title:` and be separated with `---`.

### Report:
{text}
### End of Report.
"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Only keep what's after the "### End of Report." marker
    if "### End of Report." in full_output:
        response = full_output.split("### End of Report.")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()

    # Parse separate YAML rules
    rules = re.split(r"(?m)^---\s*$|(?=^title:\s)", response)
    return [rule.strip() for rule in rules if rule.strip() and "title:" in rule]

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python custom.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n🔍 Analyzing: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    text = text[:8000]  # Limit to first 8000 characters to avoid overflowing context window


    print("\n🧠 Generating Sigma rules...")
    rules = generate_sigma_rules(text)

    if not rules:
        print("❌ No Sigma rules were generated. Try reducing prompt size or improving training.")
    else:
        for i, rule in enumerate(rules, 1):
            print(f"\n--- Sigma Rule #{i} ---\n{rule}")
