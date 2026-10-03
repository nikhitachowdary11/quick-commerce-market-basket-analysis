import pandas as pd

orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
order_products = pd.read_csv("data/order_products__prior.csv")

print("\nORDERS")
print(orders.shape)
print(orders.columns.tolist())
print(orders.head())

print("\nPRODUCTS")
print(products.shape)
print(products.columns.tolist())
print(products.head())

print("\nORDER PRODUCTS")
print(order_products.shape)
print(order_products.columns.tolist())
print(order_products.head())