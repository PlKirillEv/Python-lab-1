# Перцептрон (лабораторная 1, трек Data Science)

Однослойный перцептрон для бинарной классификации. Ядро написано на обычных
списках и циклах `for`, без numpy. `matplotlib` нужен только для графиков.

## Структура
- `perceptron_lab/model.py` - класс `Perceptron` (`activate`, `predict`, `train`, `score`)
- `perceptron_lab/data.py` - синтетические данные и локальная копия iris
- `perceptron_lab/plots.py` - граница решения и график ошибок по эпохам
- `perceptron.py` - CLI
- `tests/` - тесты pytest, `.github/workflows/ci.yml` - CI

## Запуск
    uv sync --group dev
    uv run python perceptron.py --samples 200 --lr 0.1 --epochs 50
    uv run python perceptron.py --iris

Создаются картинки `*_boundary.png` и `*_errors.png`.

## Тесты и проверки
    uv run pytest
    uv run ruff check . && uv run ruff format --check .
    uv run mypy .
