# Changelog

## 0.18.7
- Miscellaneous internal improvements

## 0.18.6
- Properly save and log RNG state in braidfile
- Add `log_prior` state diagnostics

## 0.18.5
- Make `Metadata` public.

## 0.18.4
- Add min, max, and median number of categories in a view to State diagnostics

## 0.18.3
- Serialize and deserialize `Engine` and `Oracle` to `Metadata`

## 0.18.2
- Engine seed control works
- Fixed a bug where generating a `rv` `Mixture` distribution from a column
  would sometimes have zero-valued weights, which `rv` will not accept.

## 0.18.1
- Fix bug that caused continuous predictions to be wrong when there are
  multiple modes far apart.
