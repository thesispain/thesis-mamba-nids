import pickle
import os

CIC = "/home/T2510596/Downloads/totally fresh/thesis_final/data/cicids2017_flows.pkl"

def check():
    if not os.path.exists(CIC):
        print("File not found.")
        return
    with open(CIC, 'rb') as f:
        data = pickle.load(f)
    print(f"Keys in first item: {list(data[0].keys())}")
    # Print sample of values for non-feature keys
    for k, v in data[0].items():
        if k != "features":
            print(f"  {k}: {v}")

    # Check valid attack categories
    cats = set()
    for d in data:
        if 'attack_cat' in d: cats.add(d['attack_cat'])
        elif 'label_str' in d: cats.add(d['label_str'])
    print(f"Unique Categories found: {cats}")

check()
