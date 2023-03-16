# References

## Bayesian statistics and information theory

For an introduction to Bayesian statistics, information theory, and Markov
chain Monte Carlo (MCMC), David MacKay's "Information Theory, Inference and
Learning Algorithms" [^mackay] is an excellent choice and it's available for
free.

[^mackay]: MacKay, D. J., & Mac Kay, D. J. (2003). Information theory,
  inference and learning algorithms. Cambridge university press.
  [(PDF)](http://www.inference.org.uk/itprnn/book.pdf)

## Dirichlet process mixture models

For an introduction to infinite mixture models via the Dirichlet process, Carl
Rasumssen's "*The infinite Gaussian mixture model*"[^rasumssen]  provides an
introduction to the model; and Radford Neal's "*Markov chain sampling methods
for Dirichlet process mixture models*"[^neal-dpmm]  provides an introduction to
basic MCMC methods. When I was learning Dirichlet process mixture models, I
found Frank Wood and Michael Black's "*A nonparametric Bayesian alternative to
spike sorting*" [^wood-spike]  extremely helpful. Because its target audience
is applied scientists it lays things out more simply and completely than a
manuscript aimed at statisticians or computer scientists might.

[^rasumssen]: Rasmussen, C. (1999). The infinite Gaussian mixture model.
  Advances in neural information processing systems, 12.
  ([PDF](https://openresearch.surrey.ac.uk/esploro/fulltext/journalArticle/Probability-density-estimation-via-an-infinite/99515730602346?repId=12139874790002346&mId=13140644290002346&institution=44SUR_INST))

  [^neal-dpmm]: Neal, R. M. (2000). Markov chain sampling methods for Dirichlet
  process mixture models. Journal of computational and graphical statistics,
  9(2), 249-265.
  ([PDF](https://www.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/Neal2000b.pdf))

  [^wood-spike]: Wood, F., & Black, M. J. (2008). A nonparametric Bayesian
  alternative to spike sorting. Journal of neuroscience methods, 173(1), 1-12.
  [(PDF)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7e3bb0d7af52a06455e52b082cc080374c1cc7f6)

## Probabilistic cross-categorization (PCC)

For a compact explanation designed for people unfamiliar with Bayesian
statistics, see Shafto, et al [^shafto-cc]. This work is targeted at
psychologists and demonstrates PCC's power to model human cognitive
capabilities. For a incredibly in-dept overview with loads of math, use cases,
and examples, see Mansinghka et al [^pcc-jmlr].

[^shafto-cc]: Shafto, P., Kemp, C., Mansinghka, V., & Tenenbaum, J. B. (2011).
  A probabilistic model of cross-categorization. Cognition, 120(1),
  1-25.[(PDF)](http://www.charleskemp.com/papers/shaftokmt11_aprobabilisticmodelofcrosscategorization.pdf)

  [^pcc-jmlr]: Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner,
  M., & Tenenbaum, J. B. (2016). Crosscat: A fully bayesian nonparametric
  method for analyzing heterogeneous, high dimensional data.
  [(PDF)](jmlr.org/papers/volume17/11-392/11-392.pdf)
