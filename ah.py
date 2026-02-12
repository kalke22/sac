from collections import defaultdict
from itertools import combinations


def print_table(data, title):
    print(f"\n--- {title} ---")
    for itemset, count in data.items():
        print(f"{itemset}: {count}")


C = {}
L = {}


class Bucket:
    def __init__(self):
        self.address: int
        self.count: int = 0
        self.itemsets: list[tuple] = []


def generate_candidates(prev_frequent_itemsets, k):

    candidates = set()

    for itemset1, itemset2 in combinations(prev_frequent_itemsets, 2):
        union_set = set(itemset1).union(set(itemset2))
        if len(union_set) == k:
            candidates.add(tuple(sorted(union_set)))

    return sorted(list(candidates))


def count_candidates(candidates, transactions):

    candidate_count = defaultdict(int)

    for candidate in candidates:
        for transaction in transactions.values():
            if all(item in transaction for item in candidate):
                candidate_count[candidate] += 1

    return candidate_count


def prune_candidates(candidate_count, min_support, prev_freq_itemsets=None):

    filtered_candidates = {}

    for itemset, count in candidate_count.items():

        if count >= min_support:
            if prev_freq_itemsets is None or len(itemset) == 1:
                filtered_candidates[itemset] = count
            else:
                subsets = combinations(itemset, len(itemset) - 1)
                if all(
                    tuple(sorted(subset)) in prev_freq_itemsets for subset in subsets
                ):
                    filtered_candidates[itemset] = count

    return filtered_candidates


def apriori(transactions, min_support):

    items = sorted(
        set(item for transaction in transactions.values() for item in transaction)
    )
    c1_list = [(item,) for item in items]

    C[1] = count_candidates(c1_list, transactions)
    L[1] = prune_candidates(C[1], min_support)

    print_table(C[1], "Candidate 1-itemsets (C1)")
    print_table(L[1], "Frequent 1-itemsets (L1)")

    k = 2

    transactions_combinations = {
        transaction_id: combinations(items, k)
        for transaction_id, items in transactions.items()
    }

    buckets = [Bucket(addr) for addr in range(7)]

    items_list = sorted(
        set(item for transaction in transactions.values() for item in transaction)
    )
    ranks = {item: idx + 1 for idx, item in enumerate(items_list)}

    hash_fn = lambda item1, item2: (ranks[item1] * 10 + ranks[item2]) % 7

    for itemset in transactions_combinations.values():
        for item in itemset:
            item1, item2 = item[0], item[1]
            address = hash_fn(item1, item2) % 7
            buckets[address].count += 1
            buckets[address].itemsets.append(item)

    print("\n--- Hash Table Buckets ---")
    for bucket in buckets:
        print(
            f"Address: {bucket.address}, Count: {bucket.count}, Itemsets: {bucket.itemsets}"
        )

    # Filter
    L2 = {
        itemset: bucket.count
        for bucket in buckets
        for itemset in bucket.itemsets
        if bucket.count >= min_support
    }
    print_table(L2, "Frequent 2-itemsets after Hashing (L2)")

    C["2"] = generate_candidates(L[k - 1].keys(), k)
    L["2"] = L2

    k = 3

    while True:

        candidates = generate_candidates(L[k - 1].keys(), k)
        if not candidates:
            break

        C[k] = count_candidates(candidates, transactions)
        L[k] = prune_candidates(C[k], min_support, L[k - 1].keys())

        if not L[k]:
            print_table(C[k], f"Candidate {k}-itemsets (C{k})")
            print(f"\nNo frequent {k}-itemsets found. Terminating.\n\n")
            break

        print_table(C[k], f"Candidate {k}-itemsets (C{k})")
        print_table(L[k], f"Frequent {k}-itemsets (L{k})")

        k += 1


def main():

    transactions = {
        "T100": ["I1", "I2", "I5"],
        "T200": ["I2", "I4"],
        "T300": ["I2", "I3"],
        "T400": ["I1", "I2", "I4"],
        "T500": ["I1", "I3"],
        "T600": ["I2", "I3"],
        "T700": ["I1", "I3"],
        "T800": ["I1", "I2", "I3", "I5"],
        "T900": ["I1", "I2", "I3"],
    }

    min_support = 2

    apriori(transactions, min_support)


if __name__ == "__main__":
    main()
