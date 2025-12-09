import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from raug.raug.eval import test_model
from raug.raug.train import fit_model
from raug.raug.loader import get_data_loader
from raug.raug.utils.loader import get_labels_frequency

from models.models_hub import set_class_model
from benchmarks.pad20plus.image_augmentation import ImgTrainTransform, ImgEvalTransform

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def _get_dataloader(df, images_path, use_metadata, batch_size, transform, img_column, 
                        target_number_column, metadata_columns, get_metadata_fn, shuffle=True):
    
    imgs_paths = df[img_column].apply(lambda img : f'{images_path}/{img}').values
    labels = df[target_number_column].values
    meta_data = None
    if use_metadata:
        meta_data = get_metadata_fn(df, metadata_columns) if get_metadata_fn else df[metadata_columns].values
    return get_data_loader (imgs_paths, labels, meta_data, transform=transform,
                                       batch_size=batch_size, shuf=shuffle, num_workers=4, pin_memory=True)

def train_test_folder(csv_path, _folder, _lr_init, _sched_factor, _sched_min_lr, _sched_patience, _batch_size, _epochs, 
          _early_stop, _weights, _model_name, _optimizer, _save_folder, _best_metric,
          _comb_method, _comb_config, _use_meta_data, _metric_early_stop,
          img_column, target_column, target_number_column, metadata_columns, images_path, get_metadata_fn=None,
          initial_weights_path=None, img_train_transform = ImgTrainTransform(), img_eval_transform = ImgEvalTransform()):

    # Loading the csv file
    df = pd.read_csv(csv_path)

    # split train test
    val = df[df['folder'] == _folder]
    train = df[df['folder'] != _folder]

    ####################################################################################################################

    ser_lab_freq = get_labels_frequency(train, target_column, img_column)
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    print(ser_lab_freq)

    ####################################################################################################################
    print("- Loading", _model_name)
    model = set_class_model(_model_name, len(_labels_name), comb_method=_comb_method, comb_config=_comb_config)

    loss_fn = nn.CrossEntropyLoss()
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)
        loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())

    val_data_loader =_get_dataloader(val, images_path, _use_meta_data, _batch_size, img_eval_transform,
                                     img_column, target_number_column, metadata_columns, get_metadata_fn,
                                     shuffle=False)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    train_data_loader =_get_dataloader(train, images_path, _use_meta_data, _batch_size, img_train_transform,
                                       img_column, target_number_column, metadata_columns, get_metadata_fn)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)

    ####################################################################################################################


    #####################################################################################################################
    if _optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=_lr_init)
    else:
        raise NotImplementedError(f'The optimizer {_optimizer} is not implemented!')

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, 
                                                        min_lr=_sched_min_lr, patience=_sched_patience,
                                                        mode='min' if _best_metric == 'loss' else 'max')

    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)

    fit_model (model, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None, metric_early_stop=_metric_early_stop,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the validation partition
    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    print("- Saving best validation metrics...")
    test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)