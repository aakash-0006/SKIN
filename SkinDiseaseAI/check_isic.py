import os, sys
import pandas as pd

def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

for f in ("outputs/splits.csv", "outputs/combined_splits.csv"):
    if not os.path.exists(f):
        sys.exit(f"Missing file: {f}")
orig = pd.read_csv("outputs/splits.csv")
comb = pd.read_csv("outputs/combined_splits.csv")
print("combined_splits.csv columns:", comb.columns.tolist())
print(comb.head(3).to_string(), "\n")

idc = pick(comb, ["image_id", "image", "image_name", "isic_id"])
spc = pick(comb, ["split", "partition", "set"])
lbc = pick(comb, ["dx", "label", "diagnosis", "class"])
src = pick(comb, ["source", "dataset", "origin"])
if not (idc and spc):
    sys.exit("Could not find an image-id and split column. Send me the column names printed above.")
print(f"using id={idc} split={spc} label={lbc} source={src}\n")
comb[idc] = comb[idc].astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

print("Images per split:\n", comb[spc].value_counts().to_string(), "\n")
if lbc:
    print("Class counts per split:\n", comb.groupby([spc, lbc]).size().unstack(fill_value=0).to_string(), "\n")
if src:
    print("Source per split:\n", comb.groupby([spc, src]).size().unstack(fill_value=0).to_string(), "\n")

o_test = set(orig.loc[orig.split == "test", "image_id"].astype(str))
c_test = set(comb.loc[comb[spc] == "test", idc])
print(f"ORIGINAL test images: {len(o_test)}   combined test images: {len(c_test)}")
print("  original test images missing from combined test:", len(o_test - c_test))
print("  extra images in combined test (not in original):", len(c_test - o_test))
leak = o_test & set(comb.loc[comb[spc] != "test", idc])
print("  ORIGINAL TEST IMAGES FOUND IN TRAIN/VAL (leakage):", len(leak))

print("duplicate image ids inside combined file:", int(comb[idc].duplicated().sum()))
sets = {s: set(comb.loc[comb[spc] == s, idc]) for s in comb[spc].unique()}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"image overlap {names[i]} vs {names[j]}:", len(sets[names[i]] & sets[names[j]]))
for g in ("lesion_id", "patient_id"):
    if g in comb.columns:
        gs = {s: set(comb.loc[comb[spc] == s, g].dropna().astype(str)) for s in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"{g} overlap {names[i]} vs {names[j]}:", len(gs[names[i]] & gs[names[j]]))
