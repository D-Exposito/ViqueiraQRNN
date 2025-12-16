"""Module implementation of a Quantum Recurrent Neural Network model, by default the one from https://arxiv.org/abs/2310.20671
Copyright (C) 2025  Daniel Expósito, José Daniel Viqueira
"""
import sys, os
sys.path.append(os.path.abspath(".."))

from ViqueiraQRNN.circuit import CircuitQRNN
from ViqueiraQRNN.compute import GradientMethod, CostFunction
from ViqueiraQRNN.model import ModelQRNN
from ViqueiraQRNN.ansatz import AnsatzQRNN, EMCZ2, EMCZ3

# Path to access CUNQA
sys.path.append(os.getenv("HOME"))
import cunqa