import time

from sacred.observers import FileStorageObserver
from benchmarks.pad20plus.folder_experiment import ex as experiment

if __name__=="__main__":
    start_time = str(time.time()).replace('.', '')

    optimizer = 'adam'
    early_stop_metric = 'loss'
    feature_fusion_methods = [None, 'metablock', ]
    models = [ 'mobilenet-v3-small',] 

    training_info_folder = f'opt_{optimizer}_early_stop_{early_stop_metric}'
    for i, _comb_method in enumerate(feature_fusion_methods):
        for model_name in models:

            folder_name = f'{training_info_folder}/{start_time}/{_comb_method}/{model_name}'

            for folder in range(1, 6):
                save_folder = f"benchmarks/pad20plus/results/{folder_name}/folder_{str(folder)}"

                experiment.observers = []
                experiment.observers.append(FileStorageObserver.create(save_folder))
                
                config = {
                    "_use_meta_data": _comb_method is not None,
                    "_comb_method": _comb_method,
                    "_save_folder": save_folder,
                    "_folder": folder,
                    "_model_name": model_name,
                    "_sched_patience": 10,
                    "_early_stop": 15,
                    "_batch_size": 65,
                    "_optimizer": optimizer,
                    "_epochs": 100,
                    '_lr_init': 0.0001,
                    '_sched_factor': 0.1,
                    '_sched_min_lr': 1e-6,
                    '_append_observer': False,
                    '_best_metric': early_stop_metric,
                }
                experiment.run(config_updates=config)