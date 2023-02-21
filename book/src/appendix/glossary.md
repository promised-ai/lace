## Glossary

Here we list some of the terminology, including acronyms, you will encounter
when using lace.

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
