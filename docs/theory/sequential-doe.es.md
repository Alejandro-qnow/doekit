# DoE secuencial / adaptativo

El DoE clásico se enseña a menudo como un plan **de un solo tiro**. En la
práctica corres una oleada, miras los datos y decides qué correr después. Desde
**doekit 0.5** ese bucle es de primera clase — sin abandonar D/A/G-eficiencia,
SPV o potencia como lenguaje común.

## Aumentar un diseño

```python
aug = ed.augment_design(base, n_add=4, criterion="D")
```

## Proponer el siguiente lote

```python
nxt = ed.propose_next_runs(base, response=y, n_add=4, budget=16)
print(nxt.comparison.summary)
nxt.to_dict()  # schema: doekit.NextRunsProposal/1
```

## Comparar diseños

```python
cmp = ed.compare_designs(actual, aumentado)
```

## Puente a optimización bayesiana

```python
cand = ed.candidates_from_bounds([("lr", 1e-4, 1e-1)], n=200)
```

Superficie ya disponible (doekit ≥ 0.6 / 0.7): [mezcla y split-plot](mixture-and-split-plot.md)
y el agregado `ed.experiment(...)` / `Experiment` en la
[chuleta de agentes](../agents/reference.md).
