# Changelog

## 0.18.2
- Engine seed control works
- Fixed a bug where generating a `rv` `Mixture` distribution from a column
  would sometimes have zero-valued weights, which `rv` will not accept.

## 0.18.1
- Fix bug that caused continuous predictions to be wrong when there are
  multiple modes far apart.
