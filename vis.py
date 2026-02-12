
# IMPORTS


import csv
from collections import defaultdict
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# FILE INPUT & PREPROCESSING


filename = "data.csv" 


# ---------- Read File ----------

print("Reading file:", filename)

# Read transactions directly - each row is a transaction
transactions = {}

with open(filename, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, 1):
        # Remove empty strings and strip whitespace
        items = set(item.strip() for item in row if item.strip())
        if items:  # Only add non-empty transactions
            transactions[f"T{i}"] = items

print("\nOriginal Transactions:")
for tid, items in transactions.items():
    print(f"{tid}: {items}")

# Get statistics
all_items = set()
for items in transactions.values():
    all_items.update(items)

print(f"\nTotal Transactions: {len(transactions)}")
print(f"Unique Items: {sorted(all_items)}")
print(f"Number of Unique Items: {len(all_items)}")



# PARAMETERS


min_support = 2
min_conf = 0.6

print(f"\nParameters:")
print(f"  Minimum Support: {min_support}")
print(f"  Minimum Confidence: {min_conf}")



# APRIORI



def apriori(transactions, min_support):

    C, L = {}, {}

    def count(candidates):

        count = defaultdict(int)

        for cand in candidates:
            for t in transactions.values():
                if all(i in t for i in cand):
                    count[cand] += 1

        return count

    items = sorted(set(i for t in transactions.values() for i in t))

    C[1] = count([(i,) for i in items])
    L[1] = {k: v for k, v in C[1].items() if v >= min_support}

    k = 2

    while L[k - 1]:

        candidates = set()

        for a, b in combinations(L[k - 1], 2):

            union = tuple(sorted(set(a) | set(b)))

            if len(union) == k:
                candidates.add(union)

        C[k] = count(candidates)

        L[k] = {k: v for k, v in C[k].items() if v >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L



# APRIORI + HASH (PCY)



def apriori_hash(transactions, min_support):

    C, L = {}, {}

    C[1] = defaultdict(int)

    for t in transactions.values():
        for i in t:
            C[1][frozenset([i])] += 1

    L[1] = {k: v for k, v in C[1].items() if v >= min_support}

    h = lambda p, b=7: sum(sum(ord(c) for c in s) for s in p) % b

    buckets = defaultdict(int)

    for t in transactions.values():
        for p in combinations(sorted(t), 2):
            buckets[h(p)] += 1

    freq = {b for b, c in buckets.items() if c >= min_support}

    C[2] = defaultdict(int)

    for t in transactions.values():
        for p in combinations(sorted(t), 2):
            if h(p) in freq:
                C[2][frozenset(p)] += 1

    L[2] = {k: v for k, v in C[2].items() if v >= min_support}

    k = 3

    while L[2] and k <= 10:  # Add limit to prevent infinite loops

        C[k] = {
            frozenset(a | b): 0
            for a, b in combinations(L[k - 1], 2)
            if sorted(a)[: k - 2] == sorted(b)[: k - 2]
        }

        for t in transactions.values():
            for c in C[k]:
                if c.issubset(t):
                    C[k][c] += 1

        L[k] = {k: v for k, v in C[k].items() if v >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L



# ECLAT



def eclat(transactions, min_support):

    C = {1: defaultdict(set)}

    for tid, items in transactions.items():
        for i in items:
            C[1][frozenset([i])].add(tid)

    L = {1: {k: v for k, v in C[1].items() if len(v) >= min_support}}

    k = 2

    while L[k - 1]:

        prev = list(L[k - 1].keys())

        C[k] = {
            frozenset(a | b): L[k - 1][a] & L[k - 1][b]
            for i, a in enumerate(prev)
            for b in prev[i + 1 :]
            if sorted(a)[: k - 2] == sorted(b)[: k - 2]
        }

        L[k] = {k: v for k, v in C[k].items() if len(v) >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L



# ASSOCIATION RULES



def generate_rules(freq, total):

    rules = []

    # Convert all keys to frozenset and handle both count formats
    freq_fs = {}
    for k, v in freq.items():
        key = frozenset(k) if not isinstance(k, frozenset) else k
        # If v is a set (from ECLAT), convert to count
        value = len(v) if isinstance(v, set) else v
        freq_fs[key] = value

    for itemset, count in freq_fs.items():

        if len(itemset) < 2:
            continue

        for a in chain.from_iterable(
            combinations(itemset, r) for r in range(1, len(itemset))
        ):

            antecedent = frozenset(a)
            consequent = itemset - antecedent

            if not consequent:
                continue

            if antecedent not in freq_fs:
                continue

            conf = count / freq_fs[antecedent]
            sup = count / total

            if conf >= min_conf:
                rules.append((antecedent, consequent, sup, conf))

    return rules



# RUN ALGORITHMS


print("\n" + "="*60)
print("RUNNING ALGORITHMS")
print("="*60)

print("\n[1/3] Running APRIORI...")
C_ap, L_ap = apriori(transactions, min_support)
print("      ✓ Complete")

print("[2/3] Running APRIORI + HASH (PCY)...")
C_pcy, L_pcy = apriori_hash(transactions, min_support)
print("      ✓ Complete")

print("[3/3] Running ECLAT...")
C_e, L_e = eclat(transactions, min_support)
print("      ✓ Complete")



# FLATTEN FREQUENT ITEMSETS



def flatten(L):

    return {k: v for Lk in L.values() for k, v in Lk.items()}


freq_ap = flatten(L_ap)
freq_pcy = flatten(L_pcy)
freq_e = flatten(L_e)



# RULES


total = len(transactions)

print("\nGenerating Association Rules...")
rules_ap = generate_rules(freq_ap, total)
rules_pcy = generate_rules(freq_pcy, total)
rules_e = generate_rules(freq_e, total)
print("✓ Rules Generated")



# PRINT RESULTS



def print_L(L, name):

    print(f"\n{'='*60}")
    print(f"{name} - FREQUENT ITEMSETS")
    print(f"{'='*60}")

    for k, v in L.items():

        print(f"\nL{k} (Size: {k}, Count: {len(v)})")
        print("-" * 60)

        for i, count in v.items():
            # Handle both formats: sets (ECLAT) and integers
            if isinstance(count, set):
                print(f"  {str(set(i)):40} -> Support: {len(count)}")
            else:
                print(f"  {str(set(i)):40} -> Support: {count}")


print_L(L_ap, "APRIORI")
print_L(L_pcy, "APRIORI + HASH")
print_L(L_e, "ECLAT")



# PRINT RULES



def print_rules(rules, name):

    print(f"\n{'='*60}")
    print(f"{name} - ASSOCIATION RULES")
    print(f"{'='*60}")
    print(f"Total Rules: {len(rules)}\n")

    if len(rules) == 0:
        print("  No rules found with given confidence threshold")
        return

    # Sort rules by confidence (descending), then support
    rules_sorted = sorted(rules, key=lambda x: (x[3], x[2]), reverse=True)

    for i, (a, c, s, conf) in enumerate(rules_sorted, 1):
        print(f"{i:2}. {set(a)} => {set(c)}")
        print(f"    Support: {s:.3f} ({int(s*total)}/{total}), Confidence: {conf:.3f} ({conf*100:.1f}%)")


print_rules(rules_ap, "APRIORI")
print_rules(rules_pcy, "APRIORI + HASH (PCY)")
print_rules(rules_e, "ECLAT")



# VISUALIZATION


print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)


def plot_L(L, title, filename):

    k_vals = []
    counts = []

    for k, v in L.items():
        k_vals.append(k)
        counts.append(len(v))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(k_vals, counts, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Itemset Size (k)", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Frequent Itemsets", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(k_vals)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


# Plot all three algorithms
plot_L(L_ap, "Apriori Frequent Itemsets", "apriori_itemsets.png")
plot_L(L_pcy, "Apriori + Hash (PCY) Frequent Itemsets", "pcy_itemsets.png")
plot_L(L_e, "ECLAT Frequent Itemsets", "eclat_itemsets.png")



# COMPARISON VISUALIZATION


# Compare all three algorithms
k_max = max(max(L_ap.keys()), max(L_pcy.keys()), max(L_e.keys()))
k_vals = list(range(1, k_max + 1))

counts_ap = [len(L_ap.get(k, {})) for k in k_vals]
counts_pcy = [len(L_pcy.get(k, {})) for k in k_vals]
counts_e = [len(L_e.get(k, {})) for k in k_vals]

x = np.arange(len(k_vals))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, counts_ap, width, label='Apriori', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, counts_pcy, width, label='Apriori+Hash (PCY)', color='coral', alpha=0.8)
bars3 = ax.bar(x + width, counts_e, width, label='ECLAT', color='lightgreen', alpha=0.8)

ax.set_xlabel('Itemset Size (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Frequent Itemsets', fontsize=13, fontweight='bold')
ax.set_title(f'Algorithm Comparison (min_support={min_support})', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: algorithm_comparison.png")
plt.close()



# RULES VISUALIZATION


# Compare number of rules generated
rules_counts = {
    'Apriori': len(rules_ap),
    'Apriori+Hash': len(rules_pcy),
    'ECLAT': len(rules_e)
}

plt.figure(figsize=(12, 7))
bars = plt.bar(rules_counts.keys(), rules_counts.values(), 
               color=['steelblue', 'coral', 'lightgreen'], 
               edgecolor='black', alpha=0.8, width=0.6)
plt.ylabel('Number of Association Rules', fontsize=13, fontweight='bold')
plt.title(f'Association Rules Comparison\n(min_support={min_support}, min_confidence={min_conf})', 
          fontsize=15, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('rules_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rules_comparison.png")
plt.close()



# SUMMARY TABLE


print("\n" + "="*60)
print("SUMMARY")
print("="*60)

summary_data = {
    'Algorithm': ['Apriori', 'Apriori+Hash', 'ECLAT'],
    'Frequent Itemsets': [len(flatten(L_ap)), len(flatten(L_pcy)), len(flatten(L_e))],
    'Association Rules': [len(rules_ap), len(rules_pcy), len(rules_e)]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print(f"\nDataset Statistics:")
print(f"  Total Transactions: {total}")
print(f"  Unique Items: {len(all_items)}")
print(f"  Items: {sorted(all_items)}")

print(f"\nParameters:")
print(f"  Minimum Support: {min_support}")
print(f"  Minimum Confidence: {min_conf}")

print("\n" + "="*60)
print("EXECUTION COMPLETED SUCCESSFULLY ✓")
print("="*60)
