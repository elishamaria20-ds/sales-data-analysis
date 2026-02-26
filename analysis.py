import pandas as pd

# Load dataset
df = pd.read_csv("Sample - Superstore.csv",  encoding="latin1")

#Total sales
total_sales = df["Sales"].sum()
print("Total sales: ",total_sales)

#Total profit
total_profit=df["Profit"].sum()
print("Total profit: ",total_profit)

#Sales by Category
sales_by_category=df.groupby("Category")["Sales"].sum()
print("\n Sales by category: ")
print(sales_by_category)

#Profit by category
profit_by_category=df.groupby("Category")["Profit"].sum()
print("\n Profit by category: ")
print(profit_by_category)

#Top 5 states by sales
top_states=df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(5)
print("\n Top 5 states by sales: ")
print(top_states)

import matplotlib.pyplot as plt

sales_by_category.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.show()