
class PAD20Plus:
    TARGET_NUMBER_COLUMN = "diagnostic-number"
    IMAGE_COLUMN = 'img-id'
    IMAGE_SOURCE_COLUMN = 'img-src'
    PATIENT_ID = 'patient-id'
    LABELS = ['P1', 'P2', 'P3', 'P4', 'P5']
    TARGET_COLUMN = 'macroCIDDiagnostic'

    NUMERICAL_FEATURES = ['age']

    RAW_CATEGORICAL_FEATURES = ["smoke", "drink", "background_father", "background_mother", "pesticide", "gender", 
            "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", 
            "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"
    ]