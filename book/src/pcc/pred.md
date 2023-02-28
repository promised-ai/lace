# Prediction & Imputation

Prediction and imputation both involve inferring an unknown quantity. Imputation refers to inferring the value of a specific cell in our table, and prediction refers to inferring a hypothetical value. 

The arguments for impute are the coordinates of the cell. We may wish to impute the cell at row `bat` and column `furry`. The arguments for prediction are the conditions we would like to use to create the conditional distribution. We may wish to predict `furry` given `flys=True`, `brown=True`, and `fierce=False`.

## Uncertainty

Uncertainty comes from several sources (to learn more about those sources, check out [this blog post](https://redpoll.ai/blog/ml-uncertainty/)):

1. Natural noise/imprecision/variance in the data-generating process
2. Missing data and features
3. Difficulty on the part of the model to capture a prediction

Type 1 uncertainty can be captured by computing the predictive distribution variance (or entropy for categorical targets). You can also visualize the predictive distribution. Observing multi-modality (multiple peaks in the distribution) can be a good indication that you are missing valuable information.

Determining how certain the model is in its ability to capture a prediction is done by assessing the consensus among the predictive distribution emitted by each state. The more alike these distributions are, the more certain the model is in its ability to capture a prediction.

Mathematically, uncertainty is formalized as the Jensen-Shannon divergence (JSD) between the state-level predictive distributions. Uncertainty goes from 0 to 1, 0 meaning that there is only one way to model a prediction, and 1 meaning that there are many ways to model a prediction and they all completely disagree.

![Prediction uncertainty in unimodal data](prediction-uncertainty.png)

**Above.** Prediction uncertainty when predicting *Period_minutes* of a satellite in the satellites data set. Note that the uncertainty value here is driven mostly by the difference variances of the state-level predictive distributions.

Certain ignorance is when the model has zero data by which to make a prediction and instead falls back to the prior distribution. This is rare, but when it happens it will be apparent. To be as general as possible, the priors for a column's component distributions are generally much more broad than the predictive distribution, so if you see a predictive distribution that is senselessly wide and does not looks like the marginal distribution of that variable (which should follow the histogram of the data), you have a certain ignorance. The fix is to fill in the data for items similar to the one you are predicting.


