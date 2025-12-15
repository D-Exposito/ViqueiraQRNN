"""
 Title: QRNN ansatz class
 Description: class that manages ansatzes for QRNN circuits with the structure from https://arxiv.org/abs/2310.20671

Created 10/07/2025
@author: dexposito (algorithm idea: jdviqueira)
Copyright (C) 2025  Daniel Expósito, José Daniel Viqueira
"""

import os, sys
import math
import inspect
import functools
import numpy as np
from typing import  Union, Any, Optional, Callable

# path to access CUNQA
sys.path.append(os.getenv("HOME"))

from cunqa.circuit import CunqaCircuit
from cunqa.circuit.circuit import _generate_id
from cunqa.circuit.parameter import variables, Variable
from cunqa.logger import logger

class AnsatzQRNNError(Exception):
    """Exception for error during QRNN ansatz creation."""
    pass

class AnsatzQRNN:
    """
    Class to define ansatzes with the QRNN structure. Conceptually these ansatzes are divided into the Encoding and the Evolution part. 
    In our implementation both of these blocks are divided in two parts ''encoder + final_encoder'' and ''evolver + final_evolver''. 
    The reason is that the ''encoder'' and the ''evolver'' will be repeated 'repeat_encode' and 'repeat_evolution' times, respectively, 
    without having to write it out explicitly.

    The full circuit can then be obtained using :py:meth:`ViqueiraQRNN.ansatz.get_full_circuit`.
    """
    def __init__(self, nE: int, nM: int, repeat_encode: int, repeat_evolution: int, name: str = ""):

        self.nE = nE
        self.nM = nM
        self._repeat_encode = repeat_encode
        self._repeat_evolution = repeat_evolution

        if name == "":
            self.name = "ansatz" + _generate_id()  
        else:
            self.name = name

        self.full_ansatz = CunqaCircuit(self.nE + self.nM, id = self.name)
    
    def set_circuit_blocks(self, encoder: Callable, final_encoder: Callable, evolver: Callable, final_evolver: Callable):
        """
        Method for setting the functions that apply the gates for each of the blocks to `self.full_ansatz`. 
        Using user defined functions allows adding parameters cleanly by using a iterator that traverses 
        the variables - see EMCZ2 and EMCZ3 below. Syntax example:

        .. code-block:: python

            ansatz = AnsatzQRNN(...)

            def encoder_func(self):
                x = variables('x:')
                for qubit in range(self.nE):
                    self.full_ansatz.ry(param = x[1], qubit = qubit)
                    self.full_ansatz.rx(param = theta[2*i], qubit = qubit)
                    self.full_ansatz.rz(param = theta[2*i-1], qubit = qubit)

            ansatz.set_circuit_blocks(encoder_func, ...)
        """
        setattr(self.__class__, 'encoder', encoder)
        setattr(self.__class__, 'final_encoder', final_encoder)
        setattr(self.__class__, 'evolver', evolver)
        setattr(self.__class__, 'final_evolver', final_evolver)

    def get_full_circuit(self):
        """
        Combines all blocks (appropriately repeated) to generate the full ansatz. 

        Returns:
            self.full_circuit (CunqaCircuit): (nE+nM)-qubit circuit describing the whole ansatz.
        """
        for _ in range(self._repeat_encode):
            self.encoder()
        
        self.final_encoder()

        for _ in range(self._repeat_evolution):
            self.evolver()

        self.final_evolver()

    
    def __call__(self):
        if len(self.full_ansatz.instructions) == 0:
            self.get_full_circuit()
        return self.full_ansatz
         

                

#########################################################################        
#################### DEFINITION OF OUR MAIN ANSATZES ####################
#########################################################################

def EMCZ2(nE: int, nM: int, repeat_encode: int, repeat_evolution: int, time_step: int):
    """
    Creates the ansatz for the EMCZ2 QRNN algorithm. 
    A picture of the ansatz form: https://github.com/jdani98/qutims/blob/release/0.2/.images/quantum_ansatz_CZladder2p1.png
    """
    emcz2 = AnsatzQRNN(nE, nM, repeat_encode, repeat_evolution, name = "EMCZ2")

    # Specify number of variables and create them
    x_num     = nE*(repeat_encode + 1)                                        
    theta_num = 2 * (nE * repeat_encode + (nE + nM) * repeat_evolution) + nE

    x    = variables(f'x{time_step}.(:{x_num})');   emcz2.x_iter     = iter(x)    # Use iterators to add variables when creating the circuit without having to handle indexes in the variable tuple 
    beta = variables(f'theta:{theta_num}')      ;   emcz2.theta_iter = iter(beta)

    # Define functions that determine the circuit in each
    def encoder_func(self):
        for qubit in range(nE):
            self.full_ansatz.ry(param = next(self.x_iter), qubit = qubit)
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rz(param = next(self.theta_iter), qubit = qubit)

    def final_encoder_func(self):
        for qubit in range(nE):
            self.full_ansatz.ry(param = next(self.x_iter), qubit = qubit)

    def evolver_func(self):
        for qubit in range(nE+nM): 
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rz(param = next(self.theta_iter), qubit = qubit)

        for qubit in range(nE+nM-1):                                            # Entanglement ladder
            self.full_ansatz.cz(qubit, qubit + 1)

    def final_evolver_func(self):
        for qubit in range(nE):
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)

    emcz2.set_circuit_blocks(encoder_func, final_encoder_func, evolver_func, final_evolver_func)

    return emcz2




def EMCZ3(nE: int, nM: int, repeat_encode: int, repeat_evolution: int, time_step: int):
    """
    Creates the ansatz for the EMCZ2 QRNN algorithm. 
    A picture of the ansatz: https://github.com/jdani98/qutims/blob/release/0.2/.images/quantum_ansatz_CZme3.png
    """
    emcz3 = AnsatzQRNN(nE, nM, repeat_encode, repeat_evolution, name = "EMCZ3")

    # Specify number of variables and create them
    x_num     = nE*(repeat_encode + 1)                                        
    theta_num = 3 * (nE * repeat_encode + (nE + nM) * repeat_evolution + nE)

    x    = variables(f'x{time_step}(:{x_num})');   emcz3.x_iter     = iter(x)    # Use iterators to add variables when creating the circuit without having to handle indexes in the variable tuple 
    beta = variables(f'theta:{theta_num}')     ;   emcz3.theta_iter = iter(beta)

    def encoder_func(self):
        for qubit in range(nE):
            self.full_ansatz.ry(param = next(self.x_iter),     qubit = qubit)
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rz(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)

    def final_encoder_func(self):
        for qubit in range(nE):
            self.full_ansatz.ry(param = next(self.x_iter), qubit = qubit)

    def evolver_func(self):
        for qubit in range(nE+nM): 
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rz(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)

        for qubit in range(nE):
            for qubit_m in range(nM):                                           # Entanglement ladder
                self.full_ansatz.cz(qubit, nE + qubit_m)
            
    def final_evolver_func(self):
        for qubit in range(nE):
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rz(param = next(self.theta_iter), qubit = qubit)
            self.full_ansatz.rx(param = next(self.theta_iter), qubit = qubit)

    emcz3.set_circuit_blocks(encoder_func, final_encoder_func, evolver_func, final_evolver_func)

    return emcz3