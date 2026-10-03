import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("Loading datasets...")

order_products = pd.read_csv(
    "data/order_products__prior.csv",
    usecols=["order_id", "product_id"]
)

products = pd.read_csv(
    "data/products.csv",
    usecols=["product_id", "product_name"]
)

# Use a manageable sample of orders
sample_order_ids = (
    order_products["order_id"]
    .drop_duplicates()
    .sample(n=10000, random_state=42)
)

sample = order_products[
    order_products["order_id"].isin(sample_order_ids)
]

# Add product names
sample = sample.merge(
    products,
    on="product_id",
    how="left"
)

print("Creating transactions...")

transactions = (
    sample.groupby("order_id")["product_name"]
    .apply(list)
    .tolist()
)

print("Number of baskets:", len(transactions))
print("Example basket:")
print(transactions[0])

# Convert transactions into one-hot encoded format
te = TransactionEncoder()

encoded = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded,
    columns=te.columns_
)

print("\nRunning FP-Growth...")

frequent_itemsets = fpgrowth(
    basket,
    min_support=0.005,
    use_colnames=True
)

frequent_itemsets = frequent_itemsets.sort_values(
    "support",
    ascending=False
)

print("\nTOP FREQUENT ITEMSETS")
print("---------------------")
print(frequent_itemsets.head(20))

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.1
)

rules = rules.sort_values(
    "lift",
    ascending=False
)

print("\nTOP ASSOCIATION RULES")
print("---------------------")

print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20)
)

# Save results
frequent_itemsets.to_csv(
    "fp_growth_itemsets.csv",
    index=False
)

rules.to_csv(
    "fp_growth_rules.csv",
    index=False
)

print("\nResults saved successfully.")

# -------------------------------------------------
# APRIORI ALGORITHM
# -------------------------------------------------

from mlxtend.frequent_patterns import apriori
import time

print("\nRunning Apriori...")

start_time = time.time()

apriori_itemsets = apriori(
    basket,
    min_support=0.005,
    use_colnames=True
)

apriori_time = time.time() - start_time

apriori_itemsets = apriori_itemsets.sort_values(
    "support",
    ascending=False
)

print("\nTOP APRIORI FREQUENT ITEMSETS")
print("-----------------------------")
print(apriori_itemsets.head(20))

# Generate association rules
apriori_rules = association_rules(
    apriori_itemsets,
    metric="confidence",
    min_threshold=0.1
)

apriori_rules = apriori_rules.sort_values(
    "lift",
    ascending=False
)

print("\nTOP APRIORI ASSOCIATION RULES")
print("-----------------------------")

print(
    apriori_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20)
)

# Save results
apriori_itemsets.to_csv(
    "apriori_itemsets.csv",
    index=False
)

apriori_rules.to_csv(
    "apriori_rules.csv",
    index=False
)

print("\nApriori execution time:", round(apriori_time, 3), "seconds")
print("Apriori results saved successfully.")

# -------------------------------------------------
# PRODUCT RECOMMENDATION SYSTEM
# -------------------------------------------------

def recommend_products(product_name, rules, top_n=5):

    recommendations = []

    for _, rule in rules.iterrows():

        antecedents = list(rule["antecedents"])
        consequents = list(rule["consequents"])

        if product_name in antecedents:

            for product in consequents:
                recommendations.append({
                    "product": product,
                    "confidence": rule["confidence"],
                    "lift": rule["lift"]
                })

    if len(recommendations) == 0:
        return pd.DataFrame()

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        ["lift", "confidence"],
        ascending=False
    )

    recommendations = recommendations.drop_duplicates(
        subset="product"
    )

    return recommendations.head(top_n)


# Example recommendation
product = "Banana"

print("\nPRODUCT RECOMMENDATIONS")
print("-----------------------")
print("Customer selected:", product)

recommendations = recommend_products(
    product,
    rules,
    top_n=5
)

if recommendations.empty:
    print("No recommendations found.")
else:
    print(recommendations.to_string(index=False))