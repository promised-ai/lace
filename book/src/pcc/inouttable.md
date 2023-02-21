# In- and out-of-table operations

In Lace there are a number of operations that seem redundant. Why is there
`simulate` and `draw`; `predict` and `impute`?  Why is there `surprisal` when
one can simple compute `-logp`? The answer is that the are in-table operations
and out-of-table (or hypothetical) operations. In-table operations use the
probability distribution at a certain cell in the PCC table, while out-of-table
operations do not take table location, and thus category and view assignments
into account. Hypothetical operations must marginalize over assignments.

Here is a table listing in-table and hypothetical operations.

| Purpose | In-table | Hypothetical |
|---------|----------|--------------|
| Draw random data | `draw` | `simulate` |
| Compute likelihood | (-) `surprisal` | `logp` |
| Find argmax of a likelihood | `impute` | `predict` |
