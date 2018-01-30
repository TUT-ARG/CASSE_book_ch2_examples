#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Example system for Chapter 2,
# "The Machine Learning Approach for Analysis of Sound Scenes and Events", Toni Heittola, Emre Cakir, and Tuomas Virtanen
# In book "Computational Analysis of Sound Scenes and Events", 2018, Springer
#
# A simple example application for single label classification, with acoustic scene classification used as example.
# Application is based on Multilayer perceptron (MLP) approach, and is implemented using Keras machine learning library.
#
# Author: Toni Heittola (toni.heittola@tut.fi)
#
# Requirements
# ============
# dcase_util >= 0.1.8
# sed_eval >= 0.2.0
# keras >= 2.0.8
#
# License
# =======
# Copyright (c) 2018 Tampere University of Technology and its licensors
# All rights reserved.
# Permission is hereby granted, without written agreement and without license or royalty fees, to use and copy the
# cassebook_example_systems (“Work”) described in T. Heittola, E. Cakir, and T. Virtanen, "The Machine Learning
# Approach for Analysis of Sound Scenes and Events", in "Computational Analysis of Sound Scenes and Events",
# Ed. Virtanen, T. and Plumbley, M. and Ellis, D., pp.13-40, 2018, Springer and composed of
# single_label_classification.py, multi_label_classification.py, and sound_event_detection.py files. This grant is
# only for experimental and non-commercial purposes, provided that the copyright notice in its entirety appear in all
# copies of this Work, and the original source of this Work, (Audio Research Group, Laboratory of Signal Processing)
# at Tampere University of Technology, is acknowledged in any publication that reports research using this Work.
# Any commercial use of the Work or any part thereof is strictly prohibited. Commercial use include, but is not
# limited to:
# - selling or reproducing the Work
# - selling or distributing the results or content achieved by use of the Work
# - providing services by using the Work.
#
# IN NO EVENT SHALL TAMPERE UNIVERSITY OF TECHNOLOGY OR ITS LICENSORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS WORK AND ITS DOCUMENTATION, EVEN IF
# TAMPERE UNIVERSITY OF TECHNOLOGY OR ITS LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# TAMPERE UNIVERSITY OF TECHNOLOGY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED
# HEREUNDER IS ON AN "AS IS" BASIS, AND THE TAMPERE UNIVERSITY OF TECHNOLOGY HAS NO OBLIGATION TO PROVIDE MAINTENANCE,
# SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#

import dcase_util
dcase_util.utils.setup_logging()

import tempfile
import numpy
import os
import sed_eval
import keras


def audio_filename_to_feature_filename(audio_filename, feature_path):
    return os.path.join(
        feature_path,
        os.path.split(audio_filename)[1].replace('.wav', '.cpickle')
    )


log = dcase_util.ui.FancyLogger()
log.title('Sound Classification / Single Label Classification (Acoustic Scene Classification Application)')

# Get dataset class and initialize it, data is stored under /tmp
db = dcase_util.datasets.TUTAcousticScenes_2017_DevelopmentSet()
db.initialize()

overwrite = False

param = dcase_util.containers.ParameterContainer({
    'flow': {
      'feature_extraction': True,
      'feature_normalization': True,
      'learning': True,
      'testing': True,
      'evaluation': True
    },
    'path': {
        'features': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'single_label_classification', 'features'),
        'normalization': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'single_label_classification', 'normalization'),
        'models': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'single_label_classification', 'models'),
        'results': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'single_label_classification', 'results'),
    },
    'feature_extraction': {
        'fs': 44100,
        'win_length_seconds': 0.04,
        'hop_length_seconds': 0.02,
        'spectrogram_type': 'magnitude',
        'window_type': 'hamming_asymmetric',
        'n_mels': 40,
        'n_fft': 2048,
        'fmin': 0,
        'fmax': 22050,
        'htk': False,
        'normalize_mel_bands': False,
    },
    'feature_aggregation': {
        'recipe': ['flatten'],
        'win_length_frames': 5,
        'hop_length_frames': 1,
    },
    'learner': {
        'validation_amount': 0.3,
        'model': [
            {
                'class_name': 'Dense',
                'config': {
                    'units': 50,
                    'kernel_initializer': 'uniform',
                    'activation': 'relu'
                }
            },
            {
                'class_name': 'Dropout',
                'config': {
                    'rate': 0.2
                }
            },
            {
                'class_name': 'Dense',
                'config': {
                    'units': 50,
                    'kernel_initializer': 'uniform',
                    'activation': 'relu'
                }
            },
            {
                'class_name': 'Dropout',
                'config': {
                    'rate': 0.2
                }
            },
            {
                'class_name': 'Dense',
                'config': {
                    'units': 'CLASS_COUNT',
                    'kernel_initializer': 'uniform',
                    'activation': 'softmax'
                }
            }
        ],
        'compile': {
            'loss': 'categorical_crossentropy',
            'metrics': ['categorical_accuracy'],
            'optimizer': keras.optimizers.Adam()
        },
        'fit': {
            'epochs': 200,
            'batch_size': 256,
            'shuffle': True,
        },
        'StopperCallback': {
            'monitor': 'val_categorical_accuracy',
            'initial_delay': 100,
            'min_delta': 0,
            'patience': 10,
        },
        'StasherCallback': {
            'monitor': 'val_categorical_accuracy',
            'initial_delay': 50,
        }
    }
})

# Make sure all paths exists
dcase_util.utils.Path().create(list(param['path'].values()))

# Feature extraction
if param.get_path('flow.feature_extraction'):
    log.section_header('Feature Extraction')

    # Prepare feature extractor
    mel_extractor = dcase_util.features.MelExtractor(**param['feature_extraction'])

    # Loop over all audio files in the dataset and extract features for them.
    for audio_filename in db.audio_files:
        log.line(os.path.split(audio_filename)[1], indent=2)

        # Get filename for feature data from audio filename
        feature_filename = audio_filename_to_feature_filename(audio_filename, param.get_path('path.features'))

        if not os.path.isfile(feature_filename) or overwrite:
            # Load audio data
            audio = dcase_util.containers.AudioContainer().load(
                filename=audio_filename,
                mono=True,
                fs=param.get_path('feature_extraction.fs')
            )

            # Extract features and store them into FeatureContainer, and save it to the disk
            features = dcase_util.containers.FeatureContainer(
                data=mel_extractor.extract(audio.data),
                time_resolution=param.get_path('feature_extraction.hop_length_seconds')
            ).save(
                filename=feature_filename
            )

    log.foot()

# Feature normalization
if param.get_path('flow.feature_normalization'):
    log.section_header('Feature Normalization')

    # Loop over all cross-validation folds and calculate mean and std for the training data
    for fold in db.folds():
        log.line('Fold {fold:d}'.format(fold=fold), indent=2)

        # Get filename for the normalization factors
        fold_stats_filename = os.path.join(param.get_path('path.normalization'), 'norm_fold_{fold:d}.cpickle'.format(fold=fold))

        if not os.path.isfile(fold_stats_filename) or overwrite:
            normalizer = dcase_util.data.Normalizer(
                filename=fold_stats_filename
            )

            # Loop through all training data
            for item in db.train(fold=fold):
                # Get feature filename
                feature_filename = audio_filename_to_feature_filename(
                    item.filename,
                    param.get_path('path.features')
                )

                # Load feature matrix
                features = dcase_util.containers.FeatureContainer().load(
                    filename=feature_filename
                )

                # Accumulate statistics
                normalizer.accumulate(features.data)

            # Finalize and save
            normalizer.finalize().save()

    log.foot()

# Learning
if param.get_path('flow.learning'):
    log.section_header('Learning')

    # Prepare feature aggregator
    aggregator = dcase_util.data.Aggregator(**param['feature_aggregation'])

    # Prepare event roller
    one_hot_encoder = dcase_util.data.OneHotEncoder(
        label_list=db.scene_labels(),
        time_resolution=param['feature_extraction']['hop_length_seconds']
    )

    # Loop over all cross-validation folds and learn acoustic models
    for fold in db.folds():
        log.line('Fold {fold:d}'.format(fold=fold), indent=2)

        # Get model filename
        fold_model_filename = os.path.join(param['path']['models'], 'model_fold_{fold:d}.h5'.format(fold=fold))

        if not os.path.isfile(fold_model_filename) or overwrite:
            # Get normalization factor filename
            fold_stats_filename = os.path.join(param['path']['normalization'], 'norm_fold_{fold:d}.cpickle'.format(fold=fold))

            # Load normalization factors
            normalizer = dcase_util.data.Normalizer().load(
                filename=fold_stats_filename
            )

            # Get validation files
            training_files, validation_files = db.validation_split(
                fold=fold,
                split_type='balanced',
                validation_amount=param.get_path('learner.validation_amount')
            )

            # Collect training and validation data
            training_X = []
            training_Y = []

            validation_X = []
            validation_Y = []
            for scene_label in db.scene_labels():
                for item in db.train(fold=fold).filter(scene_label=scene_label):
                    # Get feature filename
                    feature_filename = audio_filename_to_feature_filename(item.filename, param.get_path('path.features'))

                    # Load feature data
                    features = dcase_util.containers.FeatureContainer().load(
                        filename=feature_filename
                    )

                    # Normalize feature data
                    features.data = normalizer.normalize(
                        data=features.data
                    )

                    # Aggregate feature matrix, ie. flatten feature matrix in sliding window to capture temporal context
                    aggregator.aggregate(
                        data=features
                    )

                    # Targets
                    targets = one_hot_encoder.encode(
                        label=scene_label,
                        length_frames=features.frames
                    )

                    # Store feature data & target data
                    if item.filename in validation_files:
                        validation_X.append(features.data)
                        validation_Y.append(targets.data)

                    elif item.filename in training_files:
                        training_X.append(features.data)
                        training_Y.append(targets.data)

            training_X = numpy.hstack(training_X).T
            training_Y = numpy.hstack(training_Y).T

            validation_X = numpy.hstack(validation_X).T
            validation_Y = numpy.hstack(validation_Y).T

            keras_model = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.model'),
                input_shape=training_X.shape[1],
                constants={
                    'CLASS_COUNT': db.scene_label_count(),
                }
            )

            # Compile model
            keras_model.compile(**param.get_path('learner.compile'))

            # Show model topology
            log.line(dcase_util.keras.model_summary_string(keras_model))

            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    metric=param.get_path('learner.compile.metrics')[0],
                    loss=param.get_path('learner.compile.loss'),
                    output_type='logging'
                ),
                dcase_util.keras.StopperCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    monitor=param.get_path('learner.StopperCallback.monitor'),
                    initial_delay=param.get_path('learner.StopperCallback.initial_delay'),
                    min_delta=param.get_path('learner.StopperCallback.min_delta'),
                    patience=param.get_path('learner.StopperCallback.patience')
                ),
                dcase_util.keras.StasherCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    monitor=param.get_path('learner.StasherCallback.monitor'),
                    initial_delay = param.get_path('learner.StasherCallback.initial_delay'),
                )
            ]

            # Train model
            keras_model.fit(
                x=training_X,
                y=training_Y,
                validation_data=(validation_X, validation_Y),
                callbacks=callback_list,
                verbose=0,
                **param.get_path('learner.fit')
            )

            # Fetch best model
            for callback in callback_list:
                if isinstance(callback, dcase_util.keras.StasherCallback):
                    callback.log()
                    best_weights = callback.get_best()['weights']
                    if best_weights:
                        keras_model.set_weights(best_weights)
                    break

            # Save trained model
            keras_model.save(fold_model_filename)

    log.foot()

# Testing
if param.get_path('flow.testing'):
    log.section_header('Testing')

    # Prepare feature aggregator
    aggregator = dcase_util.data.Aggregator(**param['feature_aggregation'])

    # Loop over all cross-validation folds and test
    for fold in db.folds():
        log.line('Fold {fold:d}'.format(fold=fold), indent=2)

        # Get model filename
        fold_model_filename = os.path.join(param['path']['models'], 'model_fold_{fold:d}.h5'.format(fold=fold))

        # Initialize model to None, load when first non-tested file encountered.
        keras_model = None

        # Get normalization factor filename
        fold_stats_filename = os.path.join(param['path']['normalization'], 'norm_fold_{fold:d}.cpickle'.format(fold=fold))

        # Load normalization factors
        normalizer = dcase_util.data.Normalizer().load(
            filename=fold_stats_filename
        )

        # Get results filename
        fold_results_filename = os.path.join(param['path']['results'], 'res_fold_{fold:d}.txt'.format(fold=fold))
        if not os.path.isfile(fold_results_filename) or overwrite:
            # Load model if not yet loaded
            if not keras_model:
                import keras
                keras_model = keras.models.load_model(fold_model_filename)

            # Initialize results container
            res = dcase_util.containers.MetaDataContainer(
                filename=fold_results_filename
            )

            # Loop through all test files from the current cross-validation fold
            for item in db.test(fold=fold):
                # Get feature filename
                feature_filename = audio_filename_to_feature_filename(item.filename, param['path']['features'])

                # Load feature data
                features = dcase_util.containers.FeatureContainer().load(
                    filename=feature_filename
                )

                # Normalize feature data
                features.data = normalizer.normalize(
                    data=features.data
                )

                # Aggregate feature matrix, ie. flatten feature matrix in sliding window to capture temporal context
                aggregator.aggregate(
                    data=features
                )

                # Get network output
                probabilities = keras_model.predict(x=features.data.T).T

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type='global_threshold',
                    threshold=0.5
                )
                estimated_scene_label = dcase_util.data.DecisionEncoder(
                    label_list=db.scene_labels()
                ).majority_vote(
                    frame_decisions=frame_decisions
                )

                # Store result into results container
                res.append(
                    {
                        'filename': item.filename,
                        'scene_label': estimated_scene_label
                    }
                )

            # Save results container
            res.save()
    log.foot()

# Evaluation
if param.get_path('flow.evaluation'):
    log.section_header('Evaluation')
    all_res = []
    overall = []
    class_wise_results = numpy.zeros((len(db.folds()), len(db.scene_labels())))
    for fold in db.folds():
        fold_results_filename = os.path.join(param.get_path('path.results'), 'res_fold_{fold:d}.txt'.format(fold=fold))
        reference_scene_list = db.eval(fold=fold)
        for item_id, item in enumerate(reference_scene_list):
            reference_scene_list[item_id]['file'] = item.filename

        estimated_scene_list = dcase_util.containers.MetaDataContainer().load(
            filename=fold_results_filename
        )

        for item_id, item in enumerate(estimated_scene_list):
            estimated_scene_list[item_id]['file'] = item.filename

        evaluator = sed_eval.scene.SceneClassificationMetrics(
            scene_labels=db.scene_labels()
        )
        evaluator.evaluate(
            reference_scene_list=reference_scene_list,
            estimated_scene_list=estimated_scene_list
        )

        results = dcase_util.containers.DictContainer(evaluator.results())
        all_res.append(results)
        overall.append(results.get_path('overall.accuracy') * 100.0)
        class_wise_accuracy = []

        for scene_label_id, scene_label in enumerate(db.scene_labels()):
            class_wise_accuracy.append(results.get_path(['class_wise', scene_label, 'accuracy', 'accuracy']))
            class_wise_results[fold-1, scene_label_id] = results.get_path(['class_wise', scene_label, 'accuracy', 'accuracy'])

    # Form results table
    cell_data = class_wise_results
    scene_mean_accuracy = numpy.mean(cell_data, axis=0).reshape((1, -1))
    cell_data = numpy.vstack((cell_data, scene_mean_accuracy))
    fold_mean_accuracy = numpy.mean(cell_data, axis=1).reshape((-1, 1))
    cell_data = numpy.hstack((cell_data, fold_mean_accuracy))

    scene_list = db.scene_labels()
    scene_list.extend(['Average'])
    cell_data = [scene_list] + (cell_data * 100.0).tolist()

    column_headers = ['Scene']
    for fold in db.folds():
        column_headers.append('Fold {fold:d}'.format(fold=fold))

    column_headers.append('Average')

    log.table(
        cell_data=cell_data,
        column_headers=column_headers,
        column_separators=[0, 4],
        row_separators=[15],
        indent=2
    )
    log.foot()
