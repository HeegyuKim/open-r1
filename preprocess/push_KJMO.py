from datasets import load_dataset, DatasetDict

ds = load_dataset("HAERAE-HUB/HRM8K", "KSM", split="test")

KMS = ds.filter(lambda x: x["category"] == "KMS")
KJMO = ds.filter(lambda x: x["category"] == "KJMO")
KMO = ds.filter(lambda x: x["category"] == "KMO")
TQ = ds.filter(lambda x: x["category"] == "전공수학")
CSAT = ds.filter(lambda x: x["category"] == "수능/모의고사")


dd = DatasetDict({
    "KMS": KMS,
    "KJMO": KJMO,
    "KMO": KMO,
    "TQ": TQ,
    "CSAT": CSAT,
})

dd.push_to_hub("heegyu/HRM8K_KSM")