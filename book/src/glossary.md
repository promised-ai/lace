# Nomenclature

Here we list some of the terminology you will encounter when using lace. We
also attempt to provide a bit of basics on the Bayesian paradigm for users who
are more accustomed to traditional machine learning tools.

## What you need to know about Bayesian Statistics

Bayesian statistics is built around the idea of *posterior inference*. The
*posterior distribution* is the probability distribution of the parameters,
\\(\theta\\), of some model given observed data, \\(x\\). In math: \\( p(\theta
| s) \\). Per Bayes theorem, the posterior distribution can be written in terms
of other distributions,

\\[
p(\theta | s) = \frac{p(x|\theta)p(\theta)}{p(x)}.
\\]

Where \\( p(x|\theta) \\) is the *likelihood* of the observations given the
parameters of our model; \\( p(\theta) \\) is the *prior distribution*, which
defines our beliefs about the model parameters in the absence of data; and \\(
p(x) \\) is the *marginal likelihood*, which the likelihood of the data
marginalized over all possible models. Of these, the likelihood and prior are
the two distributions we're most concerned with. The marginal likelihood, which
is defined as

\\[
    p(x) = \int p(x|\theta)p(\theta) d\theta
\\]

is notoriously difficult and a lot of effort in Bayesian computation goes
toward making the marginal likelihood go away, so we won't talk about it much.

## Finite mixture models

A mixture model is a weighted sum of probabilistic distributions. Here is an example of bimodal mixture model.

<p align=center>
<img src="img/bimodal-mixture.png" alt="drawing" width="500"/>
</p>


This mixture model is defined as the sum of two Normal distributions:

\\[
p(x) = \frac{1}{2} N(x; \mu=-3, \sigma^2=1) + \frac{1}{2} N(x; \mu=3, \sigma^2=1).
\\]

In lace, we will often use the term *mixture component* to refer to an individual model within a mixture.

In general a mixture distribution has the form

\\[
p(x|\theta) = \sum_{i=1}^K w_i \, p(x|\theta_i),
\\]

where \\(K\\) is the number of mixture components, \\(w_i\\) is the \\(i^{th}\\) weight, and all weights are positive and sum to 1.

To draw a mixture model from the prior,

1. Draw the weights, \\( w \sim \text{Dirichlet}(\alpha) \\), where \\(\alpha\\) is a \\(K\\)-length vector of values in \\((0, \infty)\\). 
2. For \\(i \in \{1, ..., K\}\\) draw \\( \theta_i \sim p(\theta) \\)

As a note, we usually use one \\(\alpha\\) value repeated \\(K\\) times rather than \\(K\\) distinct values. We do not often have reason to think that any one component is more likely than the other, and reducing a vector to one value reduces the number of dimensions in our model.


## Dirichlet process mixture models (DPMM)

Suppose we don't know how many components are in our model. Given \\(N\\) data, there could be as few as 1 and as many as \\(N\\) components. To infer the number of components or categories, we place a prior on how categories are formed. One such prior is the Dirichlet process. To explain the Dirichlet process we use the Chinese restaurant process (CRP) formalization.

The CRP metaphor works like this: you are on your lunch break and, as one often does, you to usual luncheon spot: a magical Chinese restaurant where the rules of reality do not apply. Today you happen to arrive a bit before open and, as such, are the first customer to be seated. There is exactly one table with exactly one seat. You sit at that table. Or was the table instantiated for you? Who knows. The next customer arrives. They have a choice. They can sit with you or they can sit at a table alone. Customers at this restaurant love to sit together (customers of interdimensional restaurants tend to be very social creatures), but the owners offer a discount to customers who instantiate new tables. Mathematically, a customer sits at a table with probability

\\[
  p(z_i = k) =
    \begin{cases}
        \frac{n_k}{N_{-i}+\alpha},  & \text{if $k$ exists} \\\\
        \frac{\alpha}{N_{-i}+\alpha}, & \text{otherwise}
    \end{cases},
\\]

where \\(z_i\\) is the table of customer i, \\(n_k\\) is the number of customers currently seated at table \\(k\\), and \\(N_{-i}\\) is the total number of seated customers, not including customer i (who is still deciding where to sit).

Under the CRP formalism, we make inferences about what datum belongs to which category. The weight vector is implicit. That's it. For information on how inference is done in DPMMs check out the [literature recommendations](#literature-recommendations).

## Literature recommendations

### Bayesian statistics and information theory
For an introduction to Bayesian statistics, information theory, and Markov chain Monte Carlo (MCMC), David MacKay's "Information Theory, Inference and Learning Algorithms" [^mackay] is an excellent choice and it's available for free.

[^mackay]: MacKay, D. J., & Mac Kay, D. J. (2003). Information theory, inference and learning algorithms. Cambridge university press. [(PDF)](http://www.inference.org.uk/itprnn/book.pdf)

### Dirichlet process mixture models

For an introduction to infinite mixture models via the Dirichlet process, Carl Rasumssen's "*The infinite Gaussian mixture model*"[^rasumssen]  provides an introduction to the model; and Radford Neal's "*Markov chain sampling methods for Dirichlet process mixture models*"[^neal-dpmm]  provides an introduction to basic MCMC methods. When I was learning Dirichlet process mixture models, I found Frank Wood and Michael Black's "*A nonparametric Bayesian alternative to spike sorting*" [^wood-spike]  extremely helpful. Because its target audience is applied scientists it lays things out more simply and completely than a manuscript aimed at statisticians or computer scientists might.

[^rasumssen]: Rasmussen, C. (1999). The infinite Gaussian mixture model. Advances in neural information processing systems, 12. ([PDF](https://openresearch.surrey.ac.uk/esploro/fulltext/journalArticle/Probability-density-estimation-via-an-infinite/99515730602346?repId=12139874790002346&mId=13140644290002346&institution=44SUR_INST))

[^neal-dpmm]: Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265. ([PDF](https://www.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/Neal2000b.pdf))

[^wood-spike]: Wood, F., & Black, M. J. (2008). A nonparametric Bayesian alternative to spike sorting. Journal of neuroscience methods, 173(1), 1-12. [(PDF)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7e3bb0d7af52a06455e52b082cc080374c1cc7f6)

### Probabilistic cross-categorization (PCC)

For a compact explanation designed for people unfamiliar with Bayesian statistics, see Shafto, et al [^shafto-cc]. This work is targeted at psychologists and demonstrates PCC's power to model human cognitive capabilities. For a incredibly in-dept overview with loads of math, use cases, and examples, see Mansinghka et al [^pcc-jmlr].

[^shafto-cc]: Shafto, P., Kemp, C., Mansinghka, V., & Tenenbaum, J. B. (2011). A probabilistic model of cross-categorization. Cognition, 120(1), 1-25.[(PDF)](http://www.charleskemp.com/papers/shaftokmt11_aprobabilisticmodelofcrosscategorization.pdf)

[^pcc-jmlr]: Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner, M., & Tenenbaum, J. B. (2016). Crosscat: A fully bayesian nonparametric method for analyzing heterogeneous, high dimensional data. [(PDF)](jmlr.org/papers/volume17/11-392/11-392.pdf)
 

## Glossary

- **view**: A cluster of columns within a state.
- **category**: A cluster of rows within a view.
- **component model**: The probability distribution defining the model of a
    specific category in a column.
- **state**: what we call a lace posterior sample. Each state represents the
    current configuration (or state) of an independent markov chain. We
    aggregate over states to achieve estimate of likelihoods and uncertainties.
- **metadata**: A set of files from which an `Engine` may be loaded
- **prior**: A probability distribution that describe how likely certain
    hypotheses (model parameters) are before we observe any data.
- **hyperprior**: A prior distribution on prior. Allows us to admit a larger
    amount of initial uncertainty and permit a broader set of hypotheses.
- **empirical hyperprior**: A hyperprior with parameters derived from data.

- **CRP**: Chinese restaurant process
- **DPMM**: Dirichlet process mixture model
- **PCC**: Probabilistic cross-categorization
