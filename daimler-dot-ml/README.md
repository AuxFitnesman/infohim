# Daimler DOT — ML-проект (Deep Sets, многозадачность)

Проект закрывает ТЗ: свойства **партия → typical**, признаки взаимодействий, **Deep Sets** (не деревья), **GroupKFold** по `scenario_id`, два таргета, Docker, `inference.ipynb`, `predictions.csv`.

## Спецификация данных (регламент)

Участникам предоставляются три файла: `daimler_mixtures_train.csv`, `daimler_mixtures_test.csv` и `daimler_component_properties.csv`.

### `daimler_mixtures_train.csv`

Файл содержит **построчное описание состава рецептур** для изучения закономерностей и обучения моделей и включает:

- идентификатор сценария **`scenario_id`** (формат `train_N`);
- идентификатор компонента (**обезличенное название**, несущее информацию о типе компонента);
- идентификатор партии компонента;
- долю компонента в рецептуре (**в преобразованном виде**);
- параметры условий DOT: температура, время, доля биотоплива, дозировка катализатора.

Файл также содержит **целевые показатели** испытания DOT:

- степень изменения вязкости в ходе теста (**Delta Kin. Viscosity KV100 — relative | — Daimler Oxidation Test (DOT), %**);
- степень окисления (**Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), А/см**).

Каждому **`scenario_id`** соответствует **несколько строк** (по числу компонентов в рецептуре).

### `daimler_mixtures_test.csv`

Файл содержит рецептуры, на которых проводится тестирование моделей, и включает:

- идентификатор сценария **`scenario_id`** (формат `test_N`);
- идентификатор компонента (обезличенное название, несущее информацию о типе компонента);
- идентификатор партии компонента;
- долю компонента в рецептуре (в преобразованном виде);
- параметры условий DOT (температура, время, доля биотоплива, дозировка катализатора).

**Целевые показатели в этом файле отсутствуют.**

### `daimler_component_properties.csv`

Файл содержит свойства компонентов и партий в формате **«показатель — значение»**. В столбцах:

- идентификатор компонента;
- идентификатор партии компонента;
- наименование показателя;
- единица измерения показателя;
- значение показателя.

В коде проекта для строк с партией **`typical`** используются типичные значения по компоненту, если для конкретной партии из рецепта нет подходящих измерений (см. `src/features.py`).

## Структура

- `data/` — положите сюда `daimler_mixtures_train.csv`, `daimler_mixtures_test.csv`, `daimler_component_properties.csv`.
- `src/config.py` — имена колонок и формат `predictions.csv`.
- `src/features.py` — пайплайн признаков и JSON-артефакт.
- `src/model.py` — `DeepSetsMT`.
- `src/train.py`, `src/predict.py`, `src/interpret.py`, `src/validate_leakage.py`.
- `artifacts/` — после обучения: `model.pth`, `feature_pipeline.json`, `metrics.json`, `interpretation_report.json`.

## Команды (локально)

```bash
cd daimler-dot-ml
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.train
python -m src.predict
python -m src.interpret
python -m src.validate_leakage
```

Инференс-ноутбук без ручных действий:

```bash
jupyter nbconvert --to notebook --execute inference.ipynb --ExecutePreprocessor.timeout=1200
```

## Docker

Обучение (отдельный образ):

```bash
docker compose build train
docker compose run --rm train
```

Инференс (нужны уже обученные `artifacts/` на хосте):

```bash
docker compose build
docker compose run --rm predict
```

Или только сборка образа и выполнение ноутбука внутри:

```bash
docker build -t daimler-dot .
docker run --rm -v %cd%/data:/app/data:ro -v %cd%/artifacts:/app/artifacts daimler-dot
```

## Формат `predictions.csv`

Ровно 3 колонки, без дубликатов `scenario_id`, без пропусков:

`scenario_id`, `delta_kv100_rel_pct`, `oxidation_eot_acm`

При другом эталоне проверки переименуйте заголовки в `src/config.py`.

## Примечания

- Вес второго таргета в лоссе меньше (`0.32`), таргеты нормализуются по `y_std` train.
- Пересечение пар (компонент, партия) train/test возможно — это ожидаемо для реальных смесей; утечки **сценариев** между train и test нет (`validate_leakage`).
- Для отчёта по ТЗ дополните литобзор и расширенный EDA вручную в `notebooks/eda.ipynb`.
