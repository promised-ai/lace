# Preparing your data for Lace

Compared with many other machine learning tools, lace has very few requirements for data: data columns may be integer, continuous, or categorical string types; empty cells do not not need to be filled in; and the first column of the table must be the ID column and have label `ID`.

Note that for categorical columns, lace currently supports up to 256 unique values.


## Supported data types for inference

Lace supports several data types, and more can be supported (with some work).

### Continuous data

Continuous columns are modeled as mixtures of [Gaussian distributions](https://en.wikipedia.org/wiki/Normal_distribution). Find an explanation of the parameters in the [codebook](/basics/codebook.md#continuous)

### Categorical data

Continuous columns are modeled as mixtures of [categorical distributions](https://en.wikipedia.org/wiki/Categorical_distribution). Find an explanation of the parameters in the [codebook](/basics/codebook.md#categorical). 

### Count data

Support exists for a count data type, which is modeled as a mixture of [Poission distirbutions](https://en.wikipedia.org/wiki/Poisson_distribution), but there are some drawbacks, which make it best to convert the data to continuous.

- The Poisson distribution is a single parameter model so the location and variance of the mixture components cannot be controlled individually. In the Poission model, higher magnitude means higher variance.
- The hyper prior for count data is finicky and can often cause underflow/overflow errors when the underlying data do not look like Poisson distributions.

**If you use Count data do so because you kno that the underlying mixture components will be Poisson like and be sure the set the prior and unset the hyperprior in the [codebook](/basics/codebook.md)**

## Preparing your data for Lace

Lace is pretty forgiving when it comes to data. You can have missing values, string values, and numerical values all in the same table; but there are some rules that your data must follow for the platform to pick up on things. Here you will learn how to make sure that Lace understands your data properly.

### Using CSV data with Lace

Currently Lace only accepts CSV data. Formatting your data properly will help the platform understand your data.

Here are the rules:

1. Real-valued (continuous data) cells should have decimals.
2. Integer-values cells, whether count or categorical, should not have decimals.
3. Categorical data cells may be integers (up to 255) or string values
4. Missing cells should be empty
5. The index label should be 'ID'

Not following these rules will confuse the [codebook](/basics/codebook) and could cause parsing errors.

### Tips on creating valid data with pandas

When reading data from a csv, Pandas will convert integer columns with missing cells to float values since floats can represent `NaN`, which is how pandas represents missing data. You have a couple of options for saving your csv with both missing cells and properly formatted integers:

You can coerce the types to `Int64`, which is basically Int plus `NaN`, and then write to CSV.

```python
df['my_int_col'] = df['my_int_col'].astype('Int64')
df.to_csv('mydata.csv', index_label='ID')
```

If you have a lot of columns or particularly long columns, you might find it _much_ faster just to reformat as you write to the csv, in which case you can use the `float_format` option in `DataFrame.to_csv`

```python
df.to_csv('mydata.csv', index_label='ID', float_format='%g')
```
