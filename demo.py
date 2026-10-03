import pandas as pd

print("=" * 65)
print("     MARKET BASKET ANALYSIS FOR QUICK COMMERCE")
print("           Classification & Prediction")
print("=" * 65)

# -------------------------------------------------
# 1. PROJECT INFORMATION
# -------------------------------------------------

print("\nDATASET")
print("Instacart Online Grocery Shopping Dataset")
print("Total transaction records: 32,434,489")
print("Total products: 49,688")

# -------------------------------------------------
# 2. CLASSIFICATION RESULTS
# -------------------------------------------------

print("\n" + "=" * 65)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 65)

results = pd.DataFrame({
    "Algorithm": [
        "Decision Tree",
        "Random Forest",
        "Naive Bayes"
    ],
    "Accuracy": [
    0.6919375,
    0.696175,
    0.6535375
],

"Precision": [
    0.6706831773415443,
    0.6689197916218195,
    0.7304935259898667
],

"Recall": [
    0.7542,
    0.77685,
    0.4866
],

"F1 Score": [
    0.7099939986585236,
    0.718856270386564,
    0.5841098356966014
]
})

print(results.to_string(index=False))

print("\nClassification Target:")
print("reordered = 0 -> Product not reordered")
print("reordered = 1 -> Product reordered")

# -------------------------------------------------
# 3. MARKET BASKET ANALYSIS
# -------------------------------------------------

print("\n" + "=" * 65)
print("MARKET BASKET ANALYSIS")
print("=" * 65)

print("\nAlgorithms:")
print("1. FP-Growth")
print("2. Apriori")

print("\nAssociation Rule Metrics:")
print("- Support")
print("- Confidence")
print("- Lift")

# -------------------------------------------------
# 4. PRODUCT RECOMMENDATION
# -------------------------------------------------

rules = pd.read_csv("fp_growth_rules.csv")

def convert_frozenset_string(value):
    value = value.replace("frozenset({", "")
    value = value.replace("})", "")
    value = value.replace("'", "")
    return value

rules["antecedents_clean"] = rules["antecedents"].apply(
    convert_frozenset_string
)

rules["consequents_clean"] = rules["consequents"].apply(
    convert_frozenset_string
)

selected_product = "Banana"

recommendations = rules[
    rules["antecedents_clean"].str.contains(
        selected_product,
        case=False,
        na=False
    )
].copy()

recommendations = recommendations.sort_values(
    ["lift", "confidence"],
    ascending=False
)

print("\n" + "=" * 65)
print("PRODUCT RECOMMENDATION")
print("=" * 65)

print("\nCustomer selected:", selected_product)

if recommendations.empty:
    print("No recommendations available.")

else:
    output = recommendations[
        [
            "consequents_clean",
            "confidence",
            "lift"
        ]
    ].head(5)

    output = output.rename(
        columns={
            "consequents_clean": "Recommended Product",
            "confidence": "Confidence",
            "lift": "Lift"
        }
    )

    print("\nRecommended products:\n")
    print(output.to_string(index=False))

print("\n" + "=" * 65)
print("SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 65)