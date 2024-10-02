import matplotlib.pyplot as plt

# Your dictionary
data = {
    "2004": 3,
    "2005": 2,
    "2009": 2,
    "2012": 1,
    "2014": 1,
    "2015": 3,
    "2016": 22,
    "2017": 34,
    "2018": 41,
    "2019": 285,
    "2020": 632,
    "2021": 1383,
    "2022": 2165,
    "2023": 4327,
    "2024": 642,
}

# Create lists for years and values
years = list(data.keys())
values = list(data.values())

# Create the bar chart
plt.figure(figsize=(15, 8))
plt.bar(years, values)

# Customize the chart
plt.title("CVE Disclosures per Year for Cross-sight scripting", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of CVE Disclosures", fontsize=12)
plt.xticks(rotation=90)

# Add value labels on top of each bar
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center", va="bottom")

# Adjust layout and display the chart
plt.tight_layout()
plt.show()
