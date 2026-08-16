# Guía de contribución

Gracias por tu interés en `doekit`. Esta librería implementa métodos clásicos y
modernos de Diseño de Experimentos (DoE). Toda contribución debe mantener la
**fidelidad metodológica** a la literatura de referencia (p. ej. Montgomery;
Atkinson, Donev & Tobias; Jones & Nachtsheim) y documentar cualquier desviación.

## Entorno de desarrollo

```bash
uv sync --extra dev        # numpy/pandas/scipy + pytest + matplotlib + build
# o con pip:
pip install -e ".[dev,plot]"
```

## Antes de abrir un PR

1. **Tests verdes:** `uv run pytest -q`. Todo cambio de comportamiento necesita
   un test que lo cubra.
2. **Fidelidad teórica:** si tocas un constructor o criterio, valida contra la
   literatura y, cuando exista, contra propiedades numéricas conocidas
   (ortogonalidad, resolución, eficiencias, etc.).
3. **Notebooks:** deben ser *explicativos* (mostrar datos, distribuciones y
   gráficas, narrar el porqué), no meros envoltorios de llamadas. Regenera las
   salidas con:
   `uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb`.
4. **Documentación:** actualiza `README.md`, los docstrings y `docs/` si cambias
   la API pública. Añade una entrada en `CHANGELOG.md`. Sigue la
   [convención de docstrings](#convención-de-docstrings) abajo.
5. **Estilo:** el código y las salidas impresas usan ASCII (compatibilidad
   Windows); el markdown de documentación usa acentos correctos.

## Convención de docstrings

Estilo **NumPy** (el que parsea MkDocs: `docstring_style: numpy`) más secciones
de dominio DoE. Idioma: **inglés**. Sin etiqueta `docs:` — el summary + body
*son* la documentación.

Plantilla para API pública (métricas, constructores, fit, etc.):

```python
def efficiencies(design, model=None, ..., seed=None) -> dict:
    """One-line summary of what the function returns or does.

    Extended behavior: units, defaults, and assumptions the caller must know
    (e.g. metrics use coded factor units).

    Formulas
    --------
    - D-eff = 100 * det(X'X)^(1/p) / N
    - SPV(x) = N * x'(X'X)^-1 x

    Parameters
    ----------
    design : Design
        Design to evaluate.
    model : Model, optional
        Model matrix to score; taken from the design if omitted.

    Returns
    -------
    dict
        Exact keys / dtypes / NaN policy when the design is rank deficient.

    Raises
    ------
    ValueError
        When applicable (omit the section otherwise).

    Notes
    -----
    Edge cases, numerical caveats, literature pointers.

    Examples
    --------
    >>> import doekit as ed
    >>> eff = ed.efficiencies(ed.full_factorial(3), seed=0)
    >>> eff["D_efficiency"] > 90
    True
    """
```

| Sección | ¿Cuándo? |
|---------|----------|
| Summary (1 línea) | Siempre |
| Body | Siempre en API pública (unidades, supuestos) |
| `Formulas` | Cuando hay ecuación(es) que el usuario debe leer |
| `Parameters` / `Returns` | Siempre en API pública |
| `Raises` | Solo si hay errores esperables |
| `Notes` | Caveats, límites, referencias |
| `Examples` | API pública de alto uso; preferir `>>>` estable (seed si hay MC) |
| Helpers `_privados` | Una línea basta |

Orden preferido: summary → body → Formulas → Parameters → Returns → Raises →
Notes → Examples. No uses `Args:` (Google); mantén `Parameters`.

## Convención de versionado

Versionado Semántico. La versión vive en un solo lugar
(`doekit/_version.py`, `__version__`) y `pyproject.toml` la lee
dinámicamente (`[tool.hatch.version]`).

## Publicación (maintainers)

No documentar el flujo de release en `README.md` (esa página es la long
description de PyPI). Actualizar `CHANGELOG.md`, luego usar el script local
(gitignored bajo `scripts/`):

```bash
uv run --with tqdm --with python-dotenv python scripts/update_package.py          # TestPyPI
uv run --with tqdm --with python-dotenv python scripts/update_package.py --prod   # PyPI
uv run --with tqdm --with python-dotenv python scripts/update_package.py --bump   # forzar +0.0.1
```

El script limpia `dist/`, consulta el índice, hace auto-bump de patch si la
versión ya existe, valida con `twine check` y publica. Tokens: `TESTPYPI_TOKEN`
/ `PYPI_TOKEN` en `.env`. Alternativa manual: `uv build` → `uvx twine check`
→ `uv publish` (índices `testpypi` / `pypi` en `pyproject.toml`).
