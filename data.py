import pandas as pd

# Load main dataset
df = pd.read_csv("justice.csv")

# Extract unique case categories
unique_categories = df["issue_area"].dropna().unique()

# Create a new DataFrame for reference dataset
category_info = pd.DataFrame(unique_categories, columns=["issue_area"])
category_info["description"] = "No data available"
category_info["required_documents"] = "No data available"
category_info["next_steps"] = "No data available"

# Save as a new CSV file
category_info.to_csv("category_info.csv", index=False)

print("Category reference dataset created successfully!")
