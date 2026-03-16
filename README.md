# Устойчивость и робастность управления нелинейной динамической системы, идентифицированной на основе данных

Репозиторий содержит воспроизводимый pipeline для дипломной темы по **data-driven идентификации** нелинейной системы и **анализу робастной устойчивости** с/без обратной связи.

## Краткая постановка
Рассматривается система
\[
\dot x = f(x) + Bu,
\]
где для данных \((x(t_k),\dot x(t_k))\) строится модель
\[
\hat f(x)=\Theta(x)C.
\]
Далее рассчитываются residuals, оценивается детерминированная неопределенность \(\varepsilon\), вычисляется Якобиан в равновесии, решается уравнение Ляпунова и проводится численная проверка робастной устойчивости для open-loop и closed-loop (
\(u=Kx\)).

## Эталонная система
\[
\dot x_1 = -x_1 + x_2 + 0.8x_1x_2,\qquad
\dot x_2 = -2x_2 + 0.6x_1^2.
\]

## Структура проекта
```text
.
├── data/
│   └── processed/
│       ├── baseline_v1/
│       └── experiments_v2/
├── notebooks/
│   ├── 00_all_in_one_simplified.ipynb
│   ├── 01..04_*.ipynb
│   ├── 10_baseline_full_pipeline.ipynb
│   ├── 20_dictionary_and_noise_sweep.ipynb
│   └── archive/
├── results/
│   ├── baseline_v1/
│   └── experiments_v2/
├── report/
│   ├── interim_report.md
│   ├── baseline_v1/
│   └── synthetic_experiments_comparison.md
├── src/
└── tests/
```

## Установка
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Быстрый запуск
### 1) Базовый архивный эксперимент (baseline v1)
```bash
jupyter notebook notebooks/10_baseline_full_pipeline.ipynb
```
Артефакты сохраняются в:
- `results/baseline_v1/figures`
- `results/baseline_v1/metrics`
- `results/baseline_v1/tables`
- `data/processed/baseline_v1`

### 2) Сравнение словарей и шума (experiments v2)
```bash
jupyter notebook notebooks/20_dictionary_and_noise_sweep.ipynb
```
Артефакты сохраняются в:
- `results/experiments_v2/figures`
- `results/experiments_v2/metrics`
- `results/experiments_v2/tables`
- `data/processed/experiments_v2`

### 3) Проверка тестов
```bash
pytest
```

## Baseline vs Experiments v2
- **baseline_v1**: архив исходного рабочего сценария (воспроизводимый reference).
- **experiments_v2**: расширенная серия синтетических экспериментов со словарями и шумом.

## Словари в сравнении
- `linear`
- `linear_with_constant`
- `quadratic_full`
- `reduced_quadratic_no_cross`
- `reduced_cross_only`

## Уровни шума
\(\sigma \in \{0.00, 0.01, 0.03, 0.05\}\) (Gaussian noise для синтетических измерений).

## Основные выходные файлы
> Примечание: бинарные фигуры (`*.png`) не хранятся в Git и генерируются при запуске ноутбуков.

### Baseline
- `results/baseline_v1/metrics/baseline_summary.json`
- `results/baseline_v1/tables/baseline_summary.csv`
- `results/baseline_v1/figures/*.png` (генерируются локально)

### Sweep
- `results/experiments_v2/tables/dictionary_noise_summary.csv`
- `results/experiments_v2/metrics/dictionary_noise_summary.json`
- `results/experiments_v2/tables/closed_loop_summary.csv`
- `results/experiments_v2/metrics/ultimate_radius_comparison.json`

## Отчеты
- `report/interim_report.md` — общий промежуточный отчет.
- `report/baseline_v1/baseline_v1_repro.md` — архив baseline и шаги воспроизведения.
- `report/synthetic_experiments_comparison.md` — интерпретация новых сравнительных экспериментов.
