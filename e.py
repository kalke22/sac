# -------- ECLAT --------
from itertools import chain, combinations

transactions = {
    "10": {"A", "C", "D"},
    "20": {"B", "C", "E"},
    "30": {"A", "B", "C", "E"},
    "40": {"B", "E"},
}

min_support = 2
min_conf = 0.7

genL = lambda C: {k: v for k, v in C.items() if len(v) >= min_support}

C = {1: {}}
for tid, items in transactions.items():
    for i in items:
        C[1].setdefault(frozenset([i]), set()).add(tid)

L = {1: genL(C[1])}

k = 2
while L[k - 1]:
    prev = list(L[k - 1].keys())
    C[k] = {
        frozenset(a | b): L[k - 1][a] & L[k - 1][b]
        for i, a in enumerate(prev)
        for b in prev[i + 1 :]
        if sorted(a)[: k - 2] == sorted(b)[: k - 2]
    }
    L[k] = genL(C[k])
    if not L[k]:
        break
    k += 1

frequent_itemsets = {k: v for Lk in L.values() for k, v in Lk.items()}
total = len(transactions)
rules = []
for itemset, tids in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    for a in chain.from_iterable(
        combinations(itemset, r) for r in range(1, len(itemset))
    ):
        antecedent = frozenset(a)
        consequent = itemset - antecedent
        if len(consequent) == 0:
            continue
        support = len(tids) / total
        confidence = len(tids) / len(frequent_itemsets[antecedent])
        if confidence >= min_conf:
            rules.append((antecedent, consequent, support, confidence))

for k, v in C.items():
    print(f"\nC{k}:")
    print("Empty" if not v else "\n".join(f"{set(x)} : {y}" for x, y in v.items()))

for k, v in L.items():
    print(f"\nL{k}:")
    print("Empty" if not v else "\n".join(f"{set(x)} : {len(y)}" for x, y in v.items()))

print(f"\nAssociation Rules (conf >= {min_conf:.0%}):")
for a, c, s, conf in rules:
    print(f"{set(a)} => {set(c)} | support: {s:.2f}, confidence: {conf:.2f}")

