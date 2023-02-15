# Row similarity

Row similarity is (referred to in code as `rowsim`) a measurement of the similarity between rows. But row similarity is not a measurement of the distance between the values of the rows, but is a measure of *how similarly the values of two rows are modeled*. Row similarity is not a measurement in the data space, but in the model space. As such, we do not need to worry about coming up with an appropriate distance metric that incorporates data of different types, and we do not need to fret about missing data. 

Rows whose values are modeled more similarly will have higher row similarity.

The technology underlying the Lace platform clusters columns into *views*, and within each view, clusters rows into *categories*. The row similarity is the average over states of the proportion of views in a state in which the two rows are in the same category. 

\\[
RS(A, B) \frac{1}{S} \sum_{s \in S} \frac{1}{V_s}\sum_{v \in V_s} [v_a = v_b] 
\\]

Where S is the set of states, V<sub>s</sub> is the set of assignments of rows in views to categories, and v<sub>a</sub> is the assignment of row a in a particular view.

## Column-weighted variant

One may wish to weight by the size of the view. For example, if 99% of the columns are in one view, and the two rows are together in the large view, but not the small view, we would like a row similarity of 99%, not 50%. For this reason, there is a column-weighted variant, which can be accessed by way of an extra argument to the `rowsim` function.

\\[
\bar{RS}(A, B) \frac{1}{S} \sum_{s \in S} \sum_{v \in V_s} \frac{|C_v|}{|C|} [v_a = v_b] 
\\]

where C is the set of all columns in the table and C<sub>v</sub> is the number of columns in a given view, v.

We can see the effect of column weighting when computing the row similarity of animals in the zoo dataset.

![Standard row similarity for the animals data set](platform/animals-rowsim.png)

**Above.** Standard row similarity for the animals data set.

![Column-weighted row similarity for the animals data set](platform/animals-rowsim-weighted.png)

**Above.** Column-weighted row similarity for the animals data set. Note that the clusters are more pronounced.


## Contextualization

Often, we are not interested in aggregate similarity over all variables, but in similarity *with respect to* specific target variables. For example, if we are an seeds company looking to determine where certain seeds will be more effective, we might not want to compute row similarity of locations across all variables, but might be more interested in row similarity with respect to yield.

Contextualized row similarity (usually via the `wrt` [with respect to] argument) is computed only over the views containing the columns of interest. When contextualizing with a single column, column-weighted and standard row similarity are equivalent.

![Row similarity for the animals data set contextualized to 'swims'](platform/animals-rowsim-swims.png)

**Above.** Row similarity for the animals data set with respect to the *swims* variable. Animals that swim are colored blue. Animals that do not are colored tan. Note that if row similarity were looking at just the values of the data, similarity would either be zero (similar) or one (dissimilar) because the animals data are binary and we are looking it only one column. But row similarity here captures nuanced information about how *swims* is modeled. We see that withing the animals that swims, there are two distinct clusters of similarity. There are animals like the dolphin and killer whale that live their lives in the water, and there are animals like the polar bear and hippo that just visit. Both of these groups of animals swim, but for each group, Lace predicts that they swim for different reasons.


