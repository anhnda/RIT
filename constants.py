from pathlib import Path
import sys, os
from secret import MIMIC_PATH_STR, POSTGRESQL_CONNECTION_STRING
#from pandasql import PandaSQL


MIMIC_PATH = Path(MIMIC_PATH_STR)
PT = os.path.dirname(os.path.abspath(__file__))

# temporary path
TEMP_PATH = Path( "%s/data"  %PT)
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# result path 
RESULT_PATH = Path("result")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# measures whose null represent false value
NULLABLE_MEASURES = [
    "dka_type",
    "macroangiopathy",
    "microangiopathy",
    "mechanical_ventilation",
    "use_NaHCO3",
    "history_aci",
    "history_ami",
    "congestive_heart_failure",
    "liver_disease",
    "ckd_stage",
    "malignant_cancer",
    "hypertension",
    "uti",
    "chronic_pulmonary_disease",
]

# categorical values
CATEGORICAL_MEASURES = [
    "dka_type",
    "gender",
    "race",
    "liver_disease",
    "ckd_stage",
]


TARGET_PATIENT_FILE = "target_patients.csv"

queryPostgresDf = None  # set to PandaSQL(POSTGRESQL_CONNECTION_STRING) when live DB is available