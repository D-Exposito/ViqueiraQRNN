.. ViqueiraQRNN documentation master file, created by
   sphinx-quickstart on Mon Nov 10 15:40:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ViqueiraQRNN documentation
==========================

ViqueiraQRNN is a python module that implements a Quantum Recurrent Neural Network model using `CUNQA`_, a platform for the emulation of (Distributed) Quantum Computing.
QRNN models for time series prediction are customizable, for instance personalized ansatzes for the model can be created through the class ``AnsatzQRNN`` and input into the model.
Additionally, the gradient calculation method for the training can be selected from the ones available in class ``GradientMethod``. See details in the full module documentation.

.. _CUNQA: https://github.com/CESGA-Quantum-Spain/cunqa

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started

.. toctree::
   :maxdepth: 1

   VQRNN_module

   

