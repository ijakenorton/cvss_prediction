import pyjags

data_one_category = {
    # "K": 4,  # Number of subcategories
    "N": sum([80331, 2542, 26224, 1147]),  # Total observations
    "obs": [80331, 2542, 26224, 1147],  # Observations in each subcategory
    "alpha": [
        1.0,
        1.0,
        1.0,
        1.0,
    ],  # Uninformative priors for the Dirichlet distribution
}

# Model code for one category
model_code_one_category = """
model {
    pi ~ ddirch(alpha);
    obs ~ dmulti(pi, N);
}
"""

# Create and run the model
model_one_category = pyjags.Model(
    code=model_code_one_category,
    data=data_one_category,
    init=None,  # Let PyJAGS generate initial values
    chains=3,
    adapt=500,
    progress_bar=True,
)

# Burn-in phase
model_one_category.update(500)  # Burn-in period

# Sampling
samples = model_one_category.sample(1000, vars=["pi"])

# Print the sampled values for pi
print(samples["pi"])
