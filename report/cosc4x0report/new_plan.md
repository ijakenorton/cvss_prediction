# What have was I planning to do?

From an interpretability standpoint, I wanted to find out if the data had patterns within that we
could pick up with more traditional clustering means. This way it might be easier to actually be
able to interpret the data.

When looking at the data it is obvious that it is not all written in a consistent style, some of the
records added are filled with noise. Additionally there seems to be a lot of inconsistency. Though
there has been reasonable accuracies shown through LLMs and other prediction models, for those are
likely to work better if there is actual patterns within the data. Additionally we found links
between things like types of vulnerability, like denial of service will not effect data integrity.
This is probably obvious in some respects, but it is nice to see these patterns within the data.

K-means vs LDA. Used K-means as an initial look into the data, easy to setup and understand, often
the first clustering model used to see if there is anything. LDA was then used after as a method
which fits better with specifically text based modelling/clustering.

Did some initial excursions, to see if this direction seemed at least promising enough to continue. 

As this is 480 project this can go at a slower more exploratory pace.... As a result tried multiple
things to see if they made sense, the results for the clustering of NVD data using LDA looks
promising so will continue with this going forward, checking hyper-parameters etc

There have been some attempts to cluster the data, however not in the way I am doing and not with a
focus on interpretability.

# Questions to answer

Data imbalance -> did not handle to start with, but helps to interpret the data if it is balanced,
unsure if it means that have to take it with grain of salt

# Process for final results

- Run lda on balanced dataset
- For each class in each metric, find the cluster which has the best representation for that class
- Plot cluster and interpret the results
- Merge the best clusters to see if that helps

# Clustering Process 

- Think that the data should be in groups according to answers roughly

- Run kmeans on the size of the number of metrics as initial thought. Come out with some reasonable
  clusters right off the rip, reasonable in that the words make sense together

- Pursue seeing if clustering based off the different number of classes will be dominated by any of
  the classes. There is definitely a difference between the ground truth distribution and the values
  seen by the classes.

- Pivot to LDA in the hopes of a better fit and add in vector embeddings to hopefully get better
  word and word dependency representation.

- Similarly looks like LDA is showing some promising clusters / topics, but finding the data
difficult to show. Go through a few iterations, finally decide that need to dig deeper and settle on
integrityImpact as that is the most balanced of all the classes, though still decently imbalanced.
Next tests keep to a dataset that is balanced to 20000 for each class. Do various ranges of tests
with different levels of clustering. Little bit inconsistent, but manage to find models which
cluster well on the ingroup. These clusters are very strong and show definite trends, would now be
interesting to compare with one other metric to contrast and see if there is also obvious trends
there.


- Also looking into confidentialityImpact to see if there is correlation between the topics or if it
  also just has nice groups in general, this is also going to be balanced to the 20000 size to make
  the data interpretable. After all the point is to be able to see the patterns within the data. 

- Should I look for overlap between the classes? Shouldn't be too difficult, the thing is how much
does that overlap matter.



