"""
 Title: EMCZ circuit class
 Description: class that manages quantum circuits with the structure from https://arxiv.org/abs/2310.20671

Created 10/07/2025
@author: dexposito (algorithm idea: jdviqueira)
Copyright (C) 2025  Daniel Expósito, José Daniel Viqueira
"""

import os, sys
import math
import numpy as np
from typing import  Union, Any, Optional, Callable

# path to access CUNQA
sys.path.append(os.getenv("HOME"))

from cunqa.circuit import CunqaCircuit 
from cunqa.circuit.circuit import _generate_id
from cunqa.logger import logger
from cunqa.qpu import QPU
from cunqa.qjob import QJob
from ViqueiraQRNN.ansatz import AnsatzQRNN, EMCZ2, EMCZ3


class CircuitQRNNError(Exception):
    """Exception for error during QRNN circuit creation."""
    pass

class CircuitQRNN:
    def __init__(self, nE: int, nM: int, nT: int, repeat_encode: int, repeat_evolution: int, ansatz_generator: Callable[[int,int,int,int,int], AnsatzQRNN] = EMCZ2, init_state_mem: CunqaCircuit = None):
        """
        Class to manage a QRNN circuit. This circuit modifies a time series 
        to obtain another time series after executing. The default ansatz is the one from the 
        Exchange-Memory w Controlled Z-gates algorithm (https://arxiv.org/abs/2310.20671).

        Args:
            nE (int): number of qubits for the Environment/Exchange register
            nM (int): number of qubits for the Memory register
            nT (int): number of time steps of the time series
            repeat_encode (int): number of times that the encoding block should be repeated in the circuit
            repeat_evolution (int): number of times that the evolution block should be repeated in the circuit
            ansatz (int): function that produces an AnsatzQRNN instance to apply on each time step (with value dependent on the time step)
            init_state_mem (<class CunqaCircuit>): initial state for the memory register
            
        Return:
            circuit (<class CunqaCircuit>): circuit implementing the QRNN structure
        """
        self.nE = nE
        self.nM = nM
        self.nT = nT
        self._repeat_encode = repeat_encode
        self._repeat_evolution = repeat_evolution
        
        circuit_id = "CircuitQRNN_" + _generate_id()
        self.circuit = CunqaCircuit(nE + nM, nE*nT, id = circuit_id) # Num of cl_bits to measure the Environment/Exchange register on each time_step


        # Determine which ansatz to use on the circuit
        if (ansatz_generator == "EMCZ2" or ansatz_generator == EMCZ2):
            setattr(self.__class__, 'ansatz_generator', EMCZ2)
            
        elif (ansatz_generator == "EMCZ3" or ansatz_generator == EMCZ3):
            setattr(self.__class__, 'ansatz_generator', EMCZ3)
            
        else:
            
            setattr(self.__class__, 'ansatz_generator', ansatz_generator)

        # Add intial state for the memory register - if applicable
        if init_state_mem is not None:
            if init_state_mem.num_qubits != self.nM:
                logger.error(f"Initial state for the memory register has {init_state_mem.num_qubits} qubits while the memory register has {self.nM} qubits.")
                raise CircuitQRNNError

            self.init_state_mem = init_state_mem
            self.circuit += (CunqaCircuit(nE) | init_state_mem )

        ##################### BUILD THE CIRCUIT ##########################
        for time_step in range(nT):
            try:
                ansatz_obj = ansatz_generator(nE, nM, repeat_encode, repeat_evolution, time_step)
                self.circuit += ansatz_obj()
        
            except Exception as error:
                logger.error(f"An error occurred while creating the circuit:\n {error}.")
                raise CircuitQRNNError

            self.circuit.measure([i for i in range(nE)], [time_step*nE + i for i in range(nE)])
            self.circuit.save_state(label=f"State_{time_step+1}") 
            self.circuit.reset([i for i in range(nE)])         

    def bind_parameters(self, assign_dict: dict) -> None:
        """
        Method to bind values to the Variable parameters of the underlying circuit.
        Need only be used once, after submitting the circuit `QJob.upgrade_parameters()` can be used.
        """
        return self.circuit.assign_parameters(assign_dict)

    def run_on_QPU(self, qpu: QPU, **run_parameters: Any) -> QJob:
        """
        Method for running the EMC circuit on a selected QPU. 

        Args:
            QPU (class cunqa.QPU): virtual quantum processing unit where the circuit will be simulated
            **run_parameters : any other simulation instructions. For instance method="density_matrix", transpile=True, initial_layout (list with qubit layout for transpilation) 
        
        Returns:
            (class cunqa.QJob): object with the quantum simulation job. Results can be obtained doing QJob.result
        """
        try:
            qjob = qpu.run(self.circuit, **run_parameters)

        except Exception as e:
            logger.error(f"Error while running the EMCZ circuit on a QPU:\n {e}")
            raise CircuitQRNNError
        
        return qjob
          

