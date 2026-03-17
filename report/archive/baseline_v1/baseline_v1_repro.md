# Baseline v1 archive

Это архив исходного рабочего baseline-эксперимента.

## Содержимое
- `data/processed/baseline_v1/cross_coupled_dataset.csv`
- `results/archive/baseline_v1/figures/*.png (генерируются локально, не хранятся в Git)`
- `results/archive/baseline_v1/metrics/baseline_summary.json`
- `results/archive/baseline_v1/tables/baseline_summary.csv`
- `notebooks/archive/00_all_in_one_simplified_baseline_v1.ipynb`

## Как повторить
1. Установить зависимости (`pip install -r requirements.txt`).
2. Запустить `notebooks/archive/10_baseline_full_pipeline.ipynb`.
3. Проверить, что артефакты появились в `results/archive/baseline_v1` и `data/processed/baseline_v1`.

## Ключевые результаты
- Метрики идентификации и робастности сохранены в `baseline_summary.json` и `baseline_summary.csv`.
