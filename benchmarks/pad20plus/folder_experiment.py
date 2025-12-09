import time
import config
import pandas as pd

from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from benchmarks.train import train_test_folder
from benchmarks.pad20plus.image_augmentation import ImgTrainTransform, ImgEvalTransform
from models.models_hub import CONFIG_METABLOCK_BY_MODEL

TARGET_COLUMN = "macroCIDDiagnostic"
TARGET_NUMBER_COLUMN = "diagnostic-number"
IMG_COLUMN = "img-id"

_METADATA_COLUMNS = ["age", "pesticide_True", "pesticide_False", "gender_M", "gender_F", "gender_I",
                     "gender_O", "skin_cancer_history_False", "skin_cancer_history_True", "cancer_history_True",
                     "cancer_history_False", "fitspatrick_2", "fitspatrick_1", "fitspatrick_4", "fitspatrick_3",
                     "fitspatrick_6", "fitspatrick_5", "region_PEITORAL", "region_NARIZ", "region_LABIOS",
                     "region_DORSO", "region_ANTEBRACO", "region_BRACO", "region_PERNA", "region_FACE", 
                     "region_MAO", "region_COURO  CABELUDO", "region_PESCOCO", "region_PE", "region_ORELHA", 
                     "region_CABEÇA", "region_COXA", "region_ABDOME", "itch_True",
                     "itch_False", "itch_UNK", "grew_UNK", "grew_False", "grew_True", "hurt_True", "hurt_False",
                     "hurt_UNK", "changed_UNK", "changed_False", "changed_True", "bleed_False", "bleed_True",
                     "bleed_UNK", "elevation_UNK", "elevation_True", "elevation_False"]

# Starting sacred experiment
ex = Experiment()

######################################################################################

@ex.config
def cnfg():

    # Defines the folder to be used as validation
    _folder = 2

    # Models configurations
    _use_meta_data = True
    _comb_method = 'metablock' # or None
    _batch_size = 64
    _epochs = 100
    _model_name = 'mobilenet-v3-small'
    _save_folder = f"benchmarks/pad20plus/results/{_model_name}_{_comb_method}_folder_{str(_folder)}_{str(time.time()).replace('.', '')}"

    # Training variables
    _best_metric = "loss"
    _lr_init = 0.0001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None
    _weights = "frequency"
    _optimizer = 'adam'
    _append_observer = True


@ex.automain
def main (_folder, _lr_init, _sched_factor, _sched_min_lr, _sched_patience, _batch_size, _epochs, 
          _early_stop, _weights, _model_name, _save_folder, _best_metric, _optimizer,
          _comb_method, _comb_config, _use_meta_data, _metric_early_stop,_append_observer,):

    if _append_observer:
        ex.observers.append(FileStorageObserver(_save_folder))

    file_path = config.PAD_20_PLUS_ONE_HOT_ENCODED

    pd.read_csv(file_path).to_csv(Path(_save_folder).parent.parent /  'metadata.csv', index=False)

    _comb_config  = [CONFIG_METABLOCK_BY_MODEL[_model_name], len(_METADATA_COLUMNS)] if _comb_method else None,

    train_test_folder(file_path, _folder, _lr_init, _sched_factor, _sched_min_lr, 
                    _sched_patience, _batch_size, _epochs, _early_stop, _weights, _model_name,
                    _optimizer, _save_folder, _best_metric,
                    _comb_method, _comb_config, _use_meta_data, _metric_early_stop, IMG_COLUMN,
                    TARGET_COLUMN, TARGET_NUMBER_COLUMN, _METADATA_COLUMNS, config.PAD_20_PLUS_IMAGES_FOLDER,
                    initial_weights_path=None, img_train_transform=ImgTrainTransform(), img_eval_transform=ImgEvalTransform())