"""Пути и имена колонок CSV (как в выгрузке Daimler)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_CSV = "daimler_mixtures_train.csv"
TEST_CSV = "daimler_mixtures_test.csv"
PROPS_CSV = "daimler_component_properties.csv"

COL_SCENARIO = "scenario_id"
COL_COMPONENT = "Компонент"
COL_BATCH = "Наименование партии"
COL_MASS = "Массовая доля, %"
COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_DELTA = (
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
)
COL_EOT = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
COL_CAT = "Дозировка катализатора, категория"

TYPICAL_TOKEN = "typical"

# Формат сдачи: ровно 3 колонки (уточните заголовки у проверяющих при необходимости)
PRED_COL_ID = "scenario_id"
PRED_COL_DELTA = "delta_kv100_rel_pct"
PRED_COL_EOT = "oxidation_eot_acm"
