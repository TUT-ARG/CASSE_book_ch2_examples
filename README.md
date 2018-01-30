Machine Learning Approach for Analysis of Sound Scenes and Events, Examples
===========================================================================

Example systems for Chapter 2, *The Machine Learning Approach for Analysis of Sound Scenes and Events*, Toni Heittola, Emre Cakir, and Tuomas Virtanen, In book "Computational Analysis of Sound Scenes and Events", Ed. Virtanen, T. and Plumbley, M. and Ellis, D., pp.13-40, 2018, Springer 

Applications are based on multilayer perceptron (MLP) approach, and all of them are implementing similar overall system structure. Implementation is done in Python with [Keras](https://keras.io/) machine learning library.

Single-label classification
---------------------------
``single_label_classification.py``

An example system to show how to tackle single-label classification problem in audio analysis. Acoustic scene classification is used as example application with [TUT Acoustic Scenes 2017, development dataset](https://zenodo.org/record/400515#.Wm9gSXU_UeM). The setup is similar to [DCASE2017 baseline system for Task 1](https://github.com/TUT-ARG/DCASE2017-baseline-system). The dataset contains 10 second long audio excerpts from 15 different acoustic scene classes. 

Multi-label classification
--------------------------
``multi_label_classification.py``

An example system to show how to tackle multi-label classification problem in audio analysis. Audio tagging is used as example application with [CHiME-Home, development & evaluation dataset](CHiME-Home, development & evaluation dataset). 

Sound event detection
---------------------
``sound_event_detection.py``

An example system to show how to tackle detection problem in audio analysis. Sound event detection is used as example application with [TUT Sound events 2017, development dataset](https://zenodo.org/record/814831).


Getting started
===============

1. Clone repository from [Github](https://github.com/TUT-ARG/CASSE_book_ch2_examples) or download [latest release](https://github.com/TUT-ARG/CASSE_book_ch2_examples/releases/latest).
2. Install requirements with command: ``pip install -r requirements.txt``
3. Run the system: ``python single_label_classification.py``

System will download a benchmark dataset (stored under temp directory), train acoustic models based on it and evaluate system's performance. 

License
=======

Code released under the [license from Tampere University of Technology](https://github.com/TUT-ARG/CASSE_book_ch2_examples/LICENSE). Code is free to use for experimental and non-commercial purposes.