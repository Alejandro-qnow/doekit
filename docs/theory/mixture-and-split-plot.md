# Mixture and split-plot designs

From **doekit 0.6**, the advisor no longer only *warns* about mixture and
hard-to-change factors — it can shortlist real designs.

## Mixture (simplex)

Components are proportions with ``Σ x_i = 1``. Use
:class:`~doekit.domain.factors.MixtureFactor` and Scheffé models:

```python
import doekit as ed

facs = [ed.MixtureFactor("A"), ed.MixtureFactor("B"), ed.MixtureFactor("C")]
lat = ed.simplex_lattice(facs, degree=2)      # Scheffé quadratic by default
cen = ed.simplex_centroid(facs)
print(ed.evaluate(lat, n_region=2000).summary())

rec = ed.recommend_design("optimization", facs)   # or mixture=True
```

Evaluation samples a :class:`~doekit.domain.region.SimplexRegion` (Dirichlet),
not a hypercube — G-efficiency / FDS are meaningful on the simplex.

## Split-plot

Hard-to-change factors define **whole plots**; easy factors vary within plots:

```python
spd = ed.split_plot_design(
    whole_plot=[ed.ContinuousFactor("oven", 100, 200)],
    subplot=[ed.ContinuousFactor("time", 1, 10)],
    whole_plot_reps=2,
)
# Analyse:
ed.fit_mixed_model(spd, y, groups="whole_plot_id",
                   model=ed.Model.parse("0 ~ oven + time"))

rec = ed.recommend_design(
    "optimization",
    factors=[...],
    hard_to_change=["oven"],
)
```

## Constraints

```python
ed.recommend_design(
    "optimization", factors=3,
    constraints=ed.Constraints(irregular=True),   # prefers D-optimal
)
# constrained=True still works but emits DeprecationWarning
```
