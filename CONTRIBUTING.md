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
   la API pública. Añade una entrada en `CHANGELOG.md`.
5. **Estilo:** el código y las salidas impresas usan ASCII (compatibilidad
   Windows); el markdown de documentación usa acentos correctos.

## Convención de versionado

Versionado Semántico. La versión vive en un solo lugar
(`doekit/__init__.py`, `__version__`) y `pyproject.toml` la lee
dinámicamente.
