# Missing data

Lace natively handles missing data. By default Lace assumes missing data are missing completely at random, that is that the presence or absence of data is not influenced either by the value of the data in that feature (censoring) or by the value of data in other features. 

## Missing-not-at-random data in Lace

Lace can model missing-not-at-random (MNAR) data which assumes that missingness is meaningful and could be predictive of values in its feature or other features.

To use MNAR data, you can flag features as MNAR in the codebook. For example, the `longitude_radians_of_geo` column in the `Satellites` dataset is MNAR because it is only present if `Class_of_Orbit` is `GEO`.

<div class=tabbed-blocks>

```yaml
- name: longitude_radians_of_geo
  coltype: !Continuous
    hyper:
      pr_m:
        mu: 0.21544247097911842
        sigma: 1.570659039531299
      pr_k:
        shape: 1.0
        rate: 1.0
      pr_v:
        shape: 6.066108090103747
        scale: 6.066108090103747
      pr_s2:
        shape: 6.066108090103747
        scale: 2.4669698184613824
    prior: null
  notes: null
  missing_not_at_random: true
```

</div>

So if we `predict` `longitude_radians_of_geo` given different classes of orbit, we may get different values or nothing at all.

<div class=tabbed-blocks>

```python
from lace.examples import Satellites

sats = Satellites()

sats.predict(
  "longitude_radians_of_geo",
  given={"Class_of_Orbit": "GEO"}
)
# Out: (0.19146751972236603, 0.0024635919704358855)
  
sats.predict(
  "longitude_radians_of_geo",
  given={"Class_of_Orbit": "LEO"}
)
# Out: (None, 0.0009393787453276167)
```

</div>

You can also condition on `None` with MNAR columns.

<div class=tabbed-blocks>

```python
sats.predict(
  "Class_of_Orbit",
  given={"longitude_radians_of_geo": None}
))
# Out: ('LEO', 0.002252910143782927)
```

</div>

Note that conditioning on `None` will error on non-MNAR columns.

Simulate may also return `None` for MANR columns.

## How missing-not-at-random is modeled

Internally Lace models MNAR data by coupling a data feature with a binary feature that is `true` when the datum is present. This 'precense' feature is invisible to the user.
