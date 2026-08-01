# Diseños factoriales fraccionales

## Motivación

Un factorial completo en $k$ factores de dos niveles cuesta $2^k$ corridas — 1024 para
diez factores. Cuando estás dispuesto a ceder algo de información sobre interacciones de
alto orden (rara vez activas) a cambio de un recorte drástico de corridas, un
**factorial fraccional** $2^{k-p}$ corre solo una fracción cuidadosamente elegida, con
una estructura de confusión *conocida*.

## Teoría

Elige $p$ **generadores** que definan los factores extra como productos de los base,
p. ej. $D = AB$, $E = AC$. Cada generador da una **palabra** ($ABD$, $ACE$); el conjunto
de todos los productos de palabras (bajo diferencia simétrica / XOR) forma la
**relación de definición**:

$$
I = ABD = ACE = BCDE .
$$

La **resolución** $R$ es la longitud de la palabra más corta. Resume qué se confunde con
qué:

| Resolución | Efectos principales aliasados con | Lectura |
|---|---|---|
| III | interacciones de 2 factores | screening barato, riesgoso si hay 2FI activas |
| IV  | interacciones de 3 factores | principales limpios; 2FI aliasadas entre sí |
| V   | interacciones de 4 factores | principales y 2FI ambos limpios |

Los efectos aliasados comparten una **clase de alias**; estimas la suma, no los
individuos. El **folding** —anexar el diseño reflejado en signo— de-aliasa los efectos
principales de las interacciones de dos factores, subiendo la resolución.

## En doekit

```python
import doekit as ed

fr = ed.fractional_factorial(5, generators=["D=AB", "E=AC"])   # 2^(5-2) = 8 corridas
fr.metadata["defining_relation"]   # 'I = ABD = ACE = BCDE'
fr.metadata["resolution"]          # 'III'
fr.metadata["aliases"]             # clases de alias

folded = ed.fold(fr)               # de-confunde principales de 2FI
```

`fold` refleja los signos de la **matriz tal como está almacenada**. Los diseños
fraccionales y Plackett–Burman ya vienen en $\pm 1$; no pliegues un diseño cuyas
columnas sigan en unidades naturales.

## Ver también

- Teoría: [Screening (Plackett-Burman)](screening-plackett-burman.md)
- API: [`fractional_factorial`, `full_factorial`, `fold`](../api/designs.md)
