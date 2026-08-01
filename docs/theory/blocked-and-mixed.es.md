# Análisis con bloques y modelos mixtos

Desde **doekit 0.4**, el análisis ya no se limita a OLS simple. Bloques fijos,
errores estándar robustos, falta de ajuste y modelos mixtos lineales (REML) son
de primera clase, con `statsmodels` como dependencia **central**.

## Bloques fijos

Cuando un factor de ruido conocido (día, lote, operador) se controla en el
diseño, trátalo como **bloque fijo**:

```python
import doekit as ed

design = ed.attach_blocks(design, blocks=[...], name="block")
fit = ed.fit_linear_model(design, y, blocks="block")
ed.anova_table(fit)
```

`attach_blocks` escribe la columna y `metadata["blocking"]`. Si ese metadata
está presente, `fit_linear_model` lo usa automáticamente.

## Errores estándar robustos

```python
fit = ed.fit_linear_model(design, y, cov_type="HC3")
```

## Falta de ajuste

Con **réplicas** (filas idénticas de factores):

```python
lof = ed.lack_of_fit(design, y)
```

## Modelos mixtos

```python
fit = ed.fit_mixed_model(design, y, groups="batch")
```

Útil para datos ya agrupados (split-plot / multi-lote). La *generación* de
diseños split-plot sigue fuera del catálogo del asesor.

## Serialización

```python
fit.to_dict()   # schema: doekit.FitResult/1
mix.to_dict()   # schema: doekit.MixedFitResult/1
```
