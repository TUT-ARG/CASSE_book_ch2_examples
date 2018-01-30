#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Example system for Chapter 2,
# "The Machine Learning Approach for Analysis of Sound Scenes and Events", Toni Heittola, Emre Cakir, and Tuomas Virtanen
# In book "Computational Analysis of Sound Scenes and Events", 2018, Springer
#
# Simple example application for multi-label classification, with audio tagging used as example.
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
log.title('Sound Classification / Multi-Label Classification (Audio Tagging Application)')

# Get dataset class and initialize it, data is stored under /tmp
db = dcase_util.datasets.CHiMEHome_DomesticAudioTag_DevelopmentSet()
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
        'features': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'multi_label_classification', 'features'),
        'normalization': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'multi_label_classification', 'normalization'),
        'models': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'multi_label_classification', 'models'),
        'results': os.path.join(tempfile.gettempdir(), 'CASSEBOOK_CH02_Examples', 'multi_label_classification', 'results'),
    },
    'feature_extraction': {
        'fs': 16000,
        'win_length_seconds': 0.04,
        'hop_length_seconds': 0.02,
        'spectrogram_type': 'magnitude',
        'window_type': 'hann_symmetric',
        'n_mels': 40,
        'n_fft': 2048,
        'fmin': 0,
        'fmax': 8000,
        'htk': True,
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
                    'activation': 'sigmoid'
                }
            }
        ],
        'compile': {
            'loss': 'binary_crossentropy',
            'metrics': ['binary_accuracy'],
            'optimizer': keras.optimizers.Adam()
        },
        'fit': {
            'epochs': 200,
            'batch_size': 512,
            'shuffle': True,
        },
        'StopperCallback': {
            'monitor': 'EER',
            'initial_delay': 100,
            'min_delta': 0,
            'patience': 10,
        },
        'StasherCallback': {
            'monitor': 'EER',
            'initial_delay': 50,
        }
    },
    'recognizer': {
        'binarization': {
          'threshold': 0.5,
        },
        'event_activity_processing': {
            'operator': 'median_filtering',
            'window_length_seconds': 0.54,     # seconds
        },
        'event_post_processing': {
            'minimum_event_length': 0.1,  # seconds
            'minimum_event_gap': 0.1  # seconds
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

        # Get filename for feature data from audio filename
        feature_filename = audio_filename_to_feature_filename(audio_filename, param.get_path('path.features'))

        if not os.path.isfile(feature_filename) or overwrite:
            log.line(os.path.split(audio_filename)[1], indent=2)

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
                feature_filename = audio_filename_to_feature_filename(item.filename, param.get_path('path.features'))

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

    # Loop over all cross-validation folds and learn acoustic models
    for fold in db.folds():
        log.line('Fold {fold:d}'.format(fold=fold), indent=2)

        # Get model filename
        fold_model_filename = os.path.join(param['path']['models'], 'model_fold_{fold:d}.h5'.format(fold=fold))

        if not os.path.isfile(fold_model_filename) or overwrite:
            # Get normalization factor filename
            fold_stats_filename = os.path.join(param.get_path('path.normalization'), 'norm_fold_{fold:d}.cpickle'.format(fold=fold))

            # Load normalization factors
            normalizer = dcase_util.data.Normalizer().load(
                filename=fold_stats_filename
            )

            # Get validation files
            training_files, validation_files = db.validation_split(
                fold=fold,
                split_type='balanced',
                validation_amount=param.get_path('learner.validation_amount'),
                verbose=True,
                iterations=10,
            )

            training_material = db.train(fold=fold)

            # Collect training and validation data
            training_X = []
            training_Y = []

            validation_X = []
            validation_Y = []

            # Loop through all training items from specific scene class
            for audio_filename in training_material.unique_files:
                # Get feature filename
                feature_filename = audio_filename_to_feature_filename(audio_filename, param.get_path('path.features'))

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

                # Get targets
                targets = dcase_util.data.ManyHotEncoder(
                    label_list=db.tags(),
                    time_resolution=features.time_resolution
                ).encode(
                    label_list=training_material.filter(filename=audio_filename)[0].tags,
                    length_frames=features.frames
                )

                # Store feature data & target data
                if audio_filename in validation_files:
                    validation_X.append(features.data)
                    validation_Y.append(targets.data)

                elif audio_filename in training_files:
                    training_X.append(features.data)
                    training_Y.append(targets.data)

            training_X = numpy.hstack(training_X).T
            training_Y = numpy.hstack(training_Y).T

            # Store validation item indexing
            validation_X_indexing = [0]
            id = 0
            for item in validation_X:
                id += item.shape[1]
                validation_X_indexing.append(id)

            validation_X = numpy.hstack(validation_X).T
            validation_Y = numpy.hstack(validation_Y).T

            keras_model = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.model'),
                input_shape=training_X.shape[1],
                constants={
                    'CLASS_COUNT': db.tag_count(),
                }
            )

            # Compile model
            keras_model.compile(**param.get_path('learner.compile'))

            # Show model topology
            log.line(dcase_util.keras.model_summary_string(keras_model))

            processing_interval = 10
            manual_update = True
            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    metric=param.get_path('learner.compile.metrics')[0],
                    loss=param.get_path('learner.compile.loss'),
                    output_type='logging',
                    manual_update=manual_update,
                    manual_update_interval=processing_interval,
                    external_metric_labels={
                        'EER': 'Equal Error Rate'
                    }
                ),
                dcase_util.keras.StopperCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    monitor=param.get_path('learner.StopperCallback.monitor'),
                    patience=param.get_path('learner.StopperCallback.patience'),
                    initial_delay=param.get_path('learner.StopperCallback.initial_delay'),
                    min_delta=param.get_path('learner.StopperCallback.min_delta'),
                    manual_update=manual_update,
                    external_metric_labels={
                        'EER': 'Equal Error Rate'
                    }
                ),
                dcase_util.keras.StasherCallback(
                    epochs=param.get_path('learner.fit.epochs'),
                    monitor=param.get_path('learner.StasherCallback.monitor'),
                    initial_delay = param.get_path('learner.StasherCallback.initial_delay'),
                    manual_update=manual_update,
                    external_metric_labels={
                        'EER': 'Equal Error Rate'
                    }
                )
            ]
            if manual_update:
                for epoch_start in range(0, param.get_path('learner.fit.epochs'), processing_interval):
                    epoch_end = epoch_start + processing_interval

                    # Make sure we have only specified amount of epochs
                    if epoch_end > param.get_path('learner.fit.epochs'):
                        epoch_end = param.get_path('learner.fit.epochs')

                    # Train model
                    keras_model.fit(
                        x=training_X,
                        y=training_Y,
                        validation_data=(validation_X, validation_Y),
                        callbacks=callback_list,
                        verbose=0,
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        batch_size=param.get_path('learner.fit.batch_size'),
                        shuffle=param.get_path('learner.fit.shuffle')
                    )

                    # Evaluate external metrics on validation data

                    # Get per frame probabilities
                    probabilities = keras_model.predict(x=validation_X)

                    # Go validation data per file and collect tag-wise probabilities
                    estimated = dcase_util.containers.ProbabilityContainer()
                    reference = dcase_util.containers.MetaDataContainer()
                    for start_id in range(0, len(validation_X_indexing) - 1):
                        item_start_id = validation_X_indexing[start_id]
                        item_end_id = validation_X_indexing[start_id + 1]

                        ref = numpy.any(validation_Y[item_start_id:item_end_id, :], axis=0)

                        item_probabilities = probabilities[item_start_id:item_end_id, :]
                        probabilities2 = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                            probabilities=item_probabilities.T,
                            operator='mean'
                        )
                        current_results = dcase_util.containers.ProbabilityContainer()
                        ref_tags = []
                        for tag_id, tag_label in enumerate(db.tags()):
                            current_results.append(
                                {
                                    'filename': str(start_id),
                                    'label': tag_label,
                                    'probability': probabilities2[tag_id]
                                }
                            )
                            if ref[tag_id]:
                                ref_tags.append(tag_label)
                        estimated += current_results
                        reference.append(
                            {
                                'filename': str(start_id),
                                'tags': ref_tags,
                            }
                        )

                    # Evaluate
                    evaluator = sed_eval.audio_tag.AudioTaggingMetrics(tags=db.tags())
                    evaluator.evaluate(
                        reference_tag_list=reference,
                        estimated_tag_probabilities=estimated
                    )
                    eer = evaluator.results_overall_metrics()['eer']['eer']
                    # Inject external metric values to the callbacks
                    for callback in callback_list:
                        if hasattr(callback, 'set_external_metric_value'):
                            callback.set_external_metric_value(
                                metric_label='EER',
                                metric_value=eer
                            )

                    # Manually update callbacks
                    for callback in callback_list:
                        if hasattr(callback, 'update'):
                            callback.update()

                    # Check we need to stop training
                    stop_training = False
                    for callback in callback_list:
                        if hasattr(callback, 'stop'):
                            if callback.stop():
                                stop_training = True

                    if stop_training:
                        # Stop the training loop
                        break

                # Manually update callbacks
                for callback in callback_list:
                    if hasattr(callback, 'close'):
                        callback.close()

            else:
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
        fold_model_filename = os.path.join(param.get_path('path.models'), 'model_fold_{fold:d}.h5'.format(fold=fold))

        # Initialize model to None, load when first non-tested file encountered.
        keras_model = None

        # Get normalization factor filename
        fold_stats_filename = os.path.join(param.get_path('path.normalization'), 'norm_fold_{fold:d}.cpickle'.format(fold=fold))

        # Load normalization factors
        normalizer = dcase_util.data.Normalizer().load(
            filename=fold_stats_filename
        )

        # Get results filename
        fold_results_filename = os.path.join(param.get_path('path.results'), 'res_fold_{fold:d}.txt'.format(fold=fold))
        if not os.path.isfile(fold_results_filename) or overwrite:
            # Load model if not yet loaded
            if not keras_model:
                import keras
                keras_model = keras.models.load_model(fold_model_filename)

            # Initialize results container
            res = dcase_util.containers.ProbabilityContainer(
                filename=fold_results_filename
            )

            # Loop through all test files from the current cross-validation fold
            for audio_filename in db.test(fold=fold).unique_files:

                # Get feature filename
                feature_filename = audio_filename_to_feature_filename(audio_filename, param.get_path('path.features'))

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

                probabilities2 = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                    probabilities=probabilities,
                    operator='mean'
                )

                current_results = dcase_util.containers.ProbabilityContainer()
                for tag_id, tag_label in enumerate(db.tags()):
                    current_results.append(
                        {
                            'filename': db.absolute_to_relative_path(audio_filename),
                            'label': tag_label,
                            'probability': probabilities2[tag_id]
                        }
                    )

                # Store current results
                res += current_results

            # Save results container
            res.save()

    log.foot()

# Evaluation
if param.get_path('flow.evaluation'):
    log.section_header('Evaluation')
    class_wise_eer = numpy.zeros((db.fold_count, db.tag_count()))
    for fold in db.folds():
        fold_results_filename = os.path.join(param.get_path('path.results'), 'res_fold_{fold:d}.txt'.format(fold=fold))

        reference = db.eval(fold=fold)
        for item_id, item in enumerate(reference):
            item.filename = db.absolute_to_relative_path(item.filename)

        estimated = dcase_util.containers.ProbabilityContainer().load(
            filename=fold_results_filename
        )
        for item_id, item in enumerate(estimated):
            item.filename = db.absolute_to_relative_path(item.filename)

        evaluator = sed_eval.audio_tag.AudioTaggingMetrics(tags=db.tags())
        evaluator.evaluate(
            reference_tag_list=reference,
            estimated_tag_probabilities=estimated
        )

        results = dcase_util.containers.DictContainer(evaluator.results())
        for tag_id, tag_label in enumerate(db.tags()):
            class_wise_eer[fold - 1, tag_id] = results['class_wise'][tag_label]['eer']['eer'] * 100

    cell_data = class_wise_eer
    tag_mean_accuracy = numpy.nanmean(cell_data, axis=0).reshape((1, -1))
    cell_data = numpy.vstack((cell_data, tag_mean_accuracy))
    fold_mean_accuracy = numpy.nanmean(cell_data, axis=1).reshape((-1, 1))
    cell_data = numpy.hstack((cell_data, fold_mean_accuracy))

    tag_list = db.tags()
    tag_list.extend(['Average'])
    cell_data = [tag_list] + cell_data.tolist()

    column_headers = ['Tag']
    for fold in db.folds():
        column_headers.append('Fold {fold:d}'.format(fold=fold))

    column_headers.append('Average')

    log.table(
        cell_data=cell_data,
        column_headers=column_headers,
        column_separators=[0, 5],
        row_separators=[7],
        indent=2
    )

    log.foot()
