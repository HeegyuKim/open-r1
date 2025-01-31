from datasets import load_dataset


ds = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train")
with open("system.txt", "w") as f:
    for line in ds:
        f.write(f"{line['system']}\n")
        break