import pandas as pd

# Load dataset
df = pd.read_csv("Sample - Superstore.csv",  encoding="latin1")

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Year"] = df["Order Date"].dt.year

yearly_sales = df.groupby("Year")["Sales"].sum()
print("\nYearly Sales:")
print(yearly_sales)
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

plt.figure()
sales_by_category.plot(kind='bar')
plt.title("Sales by Category")
plt.savefig("sales_by_category.png")
plt.show()

plt.figure()
profit_by_category.plot(kind='bar')
plt.title("Profit by category")
plt.savefig("Profit by category.png")
plt.show()

plt.figure()
top_states.plot(kind='bar')
plt.title("Top 5 states by sales")
plt.savefig("Top 5 states by sales.png")
plt.show()

plt.figure()
yearly_sales.plot(kind="line", marker="o")
plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.savefig("yearly_sales_trend.png")
plt.show()

# Profit vs Discount analysis
plt.figure()
plt.scatter(df["Discount"], df["Profit"])
plt.title("Profit vs Discount")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.savefig("profit_vs_discount.png")
plt.show()