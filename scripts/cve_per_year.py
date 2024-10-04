import matplotlib.pyplot as plt

# Your dictionary
data = {
    "1988": 2,
    "1989": 3,
    "1990": 11,
    "1991": 15,
    "1992": 14,
    "1993": 13,
    "1994": 26,
    "1995": 25,
    "1996": 75,
    "1997": 253,
    "1998": 247,
    "1999": 923,
    "2000": 1020,
    "2001": 1679,
    "2002": 2170,
    "2003": 1548,
    "2004": 2479,
    "2005": 5010,
    "2006": 6659,
    "2007": 6596,
    "2008": 5664,
    "2009": 5778,
    "2010": 4667,
    "2011": 4172,
    "2012": 5351,
    "2013": 5324,
    "2014": 8008,
    "2015": 6595,
    "2016": 6517,
    "2017": 18113,
    "2018": 18154,
    "2019": 18938,
    "2020": 19222,
    "2021": 21950,
    "2022": 26431,
    "2023": 30949,
    "2024": 5534,
}

# Create lists for years and values
years = list(data.keys())
values = list(data.values())

# Create the bar chart
plt.figure(figsize=(15, 8))
plt.bar(years, values)

# Customize the chart
plt.title("CVE Disclosures per Year", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of CVE Disclosures", fontsize=12)
plt.xticks(rotation=90)

# Add value labels on top of each bar
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center", va="bottom")

# Adjust layout and display the chart
plt.tight_layout()
plt.show()
