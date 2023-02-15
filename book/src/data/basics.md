# Preparing your data for Lace

Compared with many other machine learning tools, lace has very few requirements for data: data columns may be integer, continuous, or categorical string types; empty cells do not not need to be filled in; and the table must contain a row index column labeled `ID`.

Note that for categorical columns, lace currently supports up to 256 unique values.

## Supported data types for inference

Lace supports several data types, and more can be supported (with some work).

### Continuous data

Continuous columns are modeled as mixtures of [Gaussian distributions](https://en.wikipedia.org/wiki/Normal_distribution). Find an explanation of the parameters in the [codebook](/codebook-ref.md#continuous)

### Categorical data

Continuous columns are modeled as mixtures of [categorical distributions](https://en.wikipedia.org/wiki/Categorical_distribution). Find an explanation of the parameters in the [codebook](/codebook-ref.md#categorical). 

### Count data

Support exists for a count data type, which is modeled as a mixture of [Poission distirbutions](https://en.wikipedia.org/wiki/Poisson_distribution), but there are some drawbacks, which make it best to convert the data to continuous.

- The Poisson distribution is a single parameter model so the location and variance of the mixture components cannot be controlled individually. In the Poisson model, higher magnitude means higher variance.
- The hyper prior for count data is finicky and can often cause underflow/overflow errors when the underlying data do not look like Poisson distributions.

**If you use Count data do so because you know that the underlying mixture components will be Poisson like and be sure the set the prior and unset the hyperprior in the [codebook](/codebook-ref.md)**

## Preparing your data for Lace

Lace is pretty forgiving when it comes to data. You can have missing values, string values, and numerical values all in the same table; but there are some rules that your data must follow for the platform to pick up on things. Here you will learn how to make sure that Lace understands your data properly.

### Accepted formats

Lace currently accepts the following data formats
- CSV
- CSV.gz (gzipped CSV)
- parquet
- IPC (feather v2)
- JSON (as output by the pandas function `df.to_json('mydata.json)`)
- JSON Lines

### Using a string-based data format

Formatting your data properly will help the platform understand your data. Under the hood, Lace uses `polars` for reading data formats into a `DataFrame`. For mote information about i/o in `polars`, see [the polars API documentation](https://pola-rs.github.io/polars/py-polars/html/reference/io.html).

Here are the rules:

1. Real-valued (continuous data) cells must have decimals.
2. Integer-values cells, whether count or categorical, must not have decimals.
3. Categorical data cells may be integers (up to 255) or string values
4. In a CSV, missing cells should be empty
5. A row index is required. The index label should be 'ID'.

Not following these rules will confuse the [codebook](/basics/codebook) and could cause parsing errors.

### Tips on creating valid data with `pandas`

When reading data from a CSV, Pandas will convert integer columns with missing cells to float values since floats can represent `NaN`, which is how pandas represents missing data. You have a couple of options for saving your CSV file with both missing cells and properly formatted integers:

You can coerce the types to `Int64`, which is basically Int plus `NaN`, and then write to CSV.

```python
df['my_int_col'] = df['my_int_col'].astype('Int64')
df.to_csv('mydata.csv', index_label='ID')
```

If you have a lot of columns or particularly long columns, you might find it _much_ faster just to reformat as you write to the csv, in which case you can use the `float_format` option in `DataFrame.to_csv`

```python
df.to_csv('mydata.csv', index_label='ID', float_format='%g')
```
