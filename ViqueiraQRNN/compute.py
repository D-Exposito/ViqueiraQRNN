"""
 Title: Gradients and Cost Functions
 Description: class implementing all gradient calculation methods and cost functions of interest. All computations of the module are concentrated here.

Created 15/07/2025
@author: dexposito (algorithm idea: jdviqueira)
Copyright (C) 2025  Daniel Expósito, José Daniel Viqueira
"""

import os, sys
import math
import functools
import copy
import numpy as np
from typing import  Union, Any, Optional

# path to access CUNQA
sys.path.append(os.getenv("HOME"))

from ViqueiraQRNN.circuit import CircuitQRNN
from cunqa.logger import logger
from cunqa.qjob import QJob, gather
from cunqa.qutils import get_QPUs
from cunqa.circuit.parameter import variables

class CostFunctionError(Exception):
    """Exception for error during cost calculations."""
    pass

class CostFunction:
    """ 
    Callable class that handles all cost functions for the QRNN model, which can be picked upon instantiation 
    or initialize with deafault and update later with ~py:meth:`update_cost_function`. 
    """
    def __init__(self, choose_function: str = "rmse"):
        
        self.choice_function = choose_function

        if choose_function == "rmse":
            self.function = self.rmse
            self.deriv = self.rmse_deriv
            functools.update_wrapper(self, self.function)

        # Add more cost functions as needed

        else:
            logger.error(f"Chosen cost function is not supported: {choose_function}.")
            raise CostFunctionError

    #################### COST FUNCTIONS ####################

    def rmse(self, prediction: np.array, y_true: np.array) -> float:
        """
        Root mean squared error.

        Args:
            prediction (np.array): value predicted from a certain vector x that we want to compare with the actual result associated to x
            y_true (np.array): actual result associated to x

        Returns:
            (float): the root mean squared error of prediction with respect to y_true
        """
        if len(prediction) != len(y_true):
            logger.error(f"Lenghts of predictions and the true labels do not match. Len(prediction) = {len(prediction)} and len(y_true) = {len(y_true)}.")
            raise CostFunctionError

        return math.sqrt(np.sum(np.power(prediction - y_true, 2))/len(y_true)) 
    
    #################### DERIVATIVES ####################

    def rmse_deriv(self, prediction: np.array, y_true: np.array) -> np.array:
        """
        Derivative of the RMSE function with respect to each of its entries.

        Args:
            prediction (np.array): value where the gradient is evaluated
            y_true (np.array): point where the gradient is computed, ie gradient of f(x) = sqrt(sum_over_i((x[i] - y_true[i])**2))
        Returns:
            (np.array): gradient of the RMSE function on the point y_true evaluated on prediction
        """
        rmse = self.rmse(prediction, y_true)

        return 2*np.array([ prediction[i]-y_true[i] for i in range(len(y_true)) ])/(len(y_true)*2*rmse)
    
    #################### INTERFACE METHODS ####################

    def update_cost_function(self, new_cost_function):
        """
        Function to change the cost function to a new one.

        Args:
            new_cost_function (str): name of the new cost function to set. Must be one of the supported ones.
        """
        logger.debug(f"Changing cost function from {self.choice_function} to {new_cost_function}")
        self.__init__(new_cost_function)

    def __call__(self, *args, **kwds):
        return self.function(*args, **kwds)
    




class GradientMethodError(Exception):
    """Exception for signaling errors during gradient calculations."""
    pass

class GradientMethod:
    """ 
    Callable class that handles all methods to calculate the gradient for the QRNN training, which can be picked upon instantiation 
    or initialize with default and update later with ~py:meth:`update_gradient_method`. 
    """
    
    def __init__(self, choose_method: str = "finite_differences"):
        """
        Args:
            choose_method (str): string describing the method that should be used. Default: finite_differences
        """
        self.choice_method = choose_method
        self.qpus=get_QPUs(on_node=False) 
        self.n_qpus = len(self.qpus)

        self.supported_methods = ["finite_differences", "parameter_shift_rule", "gradient_with_bias"]

        if choose_method in ["finite_differences", "fd"]:

            logger.debug("Selected finite_differences method")
            self.method = self.finite_differences

        elif choose_method in ["parameter_shift_rule", "psr"]:

            logger.debug("Selected parameter_shift_rule method")
            self.method = self.parameter_shift_rule

        elif choose_method in ["gradient_with_bias", "bias"]:

            logger.debug("Selected gradient_with_bias method")
            self.method = self.gradient_with_bias
        
        # Add any desired gradient methods :-)

        else:
            logger.error(f"Chosen gradient method is not supported: {choose_method}.")
            raise GradientMethodError
        
        functools.update_wrapper(self, self.method)

    #################### EVALUATION ####################

    def init_evaluation(self, circuit, time_series, theta, shots: int = 2000):
        """
        First evaluation of the QRNN circuit with the given parameters, distributing shots. QJobs are initialized and the combined result 
        is stored in the attribute ``reference``.
        """
        self.circuit = circuit
        param_dict = {}
        param_dict |= {var: x*np.pi     for var, x     in zip(variables(f'x{circuit.nT}:{len(time_series)/circuit.nT}'), time_series)}
        param_dict |= {var: theta       for var, theta in zip(variables(f'theta:{len(theta)}'),                                theta)}

        logger.debug(f"Assigning parameters in compute.init_evaluation: {param_dict}") 
        self.circuit.bind_parameters(param_dict) # HARDCODED names for the params

        shots_per_qpu = shots // self.n_qpus
        remain = shots % self.n_qpus
        distr_shots = [shots_per_qpu + 1 for _ in range(remain)] + [shots_per_qpu for _ in range(self.n_qpus-remain)]

        self.qjobs = [self.circuit.run_on_QPU(qpu, shots=distr_shots[i], method="density_matrix") for i, qpu in enumerate(self.qpus)]
        results = gather(self.qjobs)

        # Combine results from all QPUs
        nE = self.circuit.nE
        obs = [calc_observable(result, nE) for result in results]
        
        self.reference = np.sum(obs, axis=0)/len(obs) # return the mean lol
        logger.debug(f"Reference is {self.reference} which is the average of {obs}.")

        return  self.reference

    def distr_shots(self, time_series, theta, shots: int = 2000):
        """
        Evaluation of the QRNN circuit with the given parameters, distributing shots. The combined result is stored in the attribute ``reference``.
        """
        shots_per_qpu = shots // self.n_qpus
        remain = shots % self.n_qpus
        distr_shots = [shots_per_qpu + 1 for _ in range(remain)] + [shots_per_qpu for _ in range(self.n_qpus-remain)]

        param_dict = {}
        param_dict |= {var: x*np.pi     for var, x     in zip(variables(f'x{self.circuit.nT}:{len(time_series)/self.circuit.nT}'), time_series)}
        param_dict |= {var: theta       for var, theta in zip(variables(f'theta:{len(theta)}'),                                theta)}

        for i, qjob in enumerate(self.qjobs):
            qjob.upgrade_parameters(param_dict, shots=distr_shots[i])
        
        results = gather(self.qjobs)

        # Combine results from all QPUs
        nE = self.circuit.nE
        obs = [calc_observable(result, nE) for result in results]
        
        self.reference = np.sum(obs, axis=0)/len(obs) # return the mean lol
        logger.debug(f"Reference is {self.reference} which is the average of {obs}.")

        return  self.reference


    #################### GRADIENTS ####################

    def finite_differences(self, circuit: CircuitQRNN, time_series: np.array, theta_now: np.array, y_true: np.array, cost_func: CostFunction, diff: Optional[float] = 1e-7, shots: int = 2000):
        """
        Finite differences method for calculating the gradient. It estimates the derivative on 
        each component of the gradient, parallelizing the calculation between QPUs.

        Args:
            circuit (<class CircuitQRNN>): the gradient of the parameters of this circuit will be computed. It is needed for the .parameters() method that creates the right parameter order
            qjobs (list[<class cunqa.qjob>]): qjob objects representing the QPUs to which we submitted circuit. Use .upgrade_parameters() on them to input the desired parameters
            time_series (np.array): information of the time series being considered
            theta_now (np.array): initial value of theta at which to calculate the gradient 
            y_true (np.array): true label that circuit should produce from time_series
            cost_func (<class CostFunction>): function that calculates the loss 
            diff (float): small difference to put on each coordinate to calculate derivatives

        Return:
            gradient (np.array): array of lenght len(theta_now) with the estimated gradient of theta
        """
        logger.debug("Entered finite differences calculation")

        n = len(theta_now)
        n_qpus = len(self.qpus)
        gradient = np.array([0.0 for _ in range(n)])
        
        # We go through the components of theta n_qpus at a time 
        for i in range(n // n_qpus):

            # Range of components for which we calculate the derivative on this loop iteration
            start = i*n_qpus
            end = (i+1)*n_qpus

            # Concurrent execution of circuits with small differences on one component
            results = gather([perturbed_i_circ(qjob, time_series, theta_now, start + self.qjobs.index(qjob), diff, shots=shots, nT=circuit.nT) for qjob in self.qjobs])
            observables = [calc_observable(res, self.circuit.nE) for res in results]

            deriv = [np.dot(cost_func.deriv(obs, y_true), (obs - self.reference)/diff) for obs in observables]
            gradient[start:end] = deriv


        # Go through the last n % n_qjobs objects 
        final_start = n-(n % n_qpus)
        final_results = gather([perturbed_i_circ(self.qjobs[i], time_series, theta_now, final_start + i, diff, shots, nT=circuit.nT) for i in range(n % n_qpus)])
        final_observables = [calc_observable(res, self.circuit.nE) for res in final_results]
        
        final_deriv = [np.dot(cost_func.deriv(obs, y_true), (obs - self.reference)/diff)  for obs in final_observables]
        gradient[final_start:n] = final_deriv

        return gradient


    def parameter_shift_rule(self, circuit: CircuitQRNN, time_series: np.array, theta_now: np.array, y_true: np.array, cost_func: CostFunction, shots: int = 2000):
        """
        Parameter shift rule method for estimating the gradient of continuous parameters of quantum circuits.
        Based on the formula grad_{theta}f(x; theta) = [f(x; theta + pi/2) + f(x; theta - pi/2)]/2 for quantum functions.

        Args:
            circuit (<class CircuitQRNN>): the gradient of the parameters of this circuit will be computed. It is needed for the .parameters() method that creates the right parameter order
            qjobs (list[<class cunqa.qjob>]): qjob objects representing the QPUs to which we submitted circuit. Use .upgrade_parameters() on them to input the desired parameters
            time_series (np.array): information of the time series being considered
            theta_now (np.array): initial value of theta at which to calculate the gradient 
            y_true (np.array): true label that circuit should produce from time_series
            cost_func (<class CostFunction>): function that calculates the loss 

        Return:
            gradient (np.array): array of lenght len(theta_now) with the estimated gradient of theta
        """
        logger.debug("Entered parameter shift rule calculation")

        n = len(theta_now)
        half_qpus = len(self.qjobs) // 2 # If num_QPUs is not even, a QPU will be wasted
        gradient = np.array([0.0 for _ in range(n)])

        # We go through the components of theta in blocks of size n_qpus/2 
        for i in range(n // half_qpus):

            # Range of components for which we calculate the derivative on this loop iteration
            start = i*half_qpus
            end = (i+1)*half_qpus

            # Concurrent execution of circuits with the parameter shifted +-pi/2 on one component
            results = gather([plus_minus for qjob_minus, qjob_plus in zip(self.qjobs[0::2], self.qjobs[1::2]) for plus_minus in shifted_i_circ(qjob_minus, qjob_plus, time_series, theta_now, start + self.qjobs[0::2].index(qjob_minus), shots=shots, nT=circuit.nT)])
            observables = [calc_observable(res, circuit.nE) for res in results]

            deriv = [
                np.dot( (cost_func.deriv(plus, y_true) - cost_func.deriv(minus, y_true))/2, (plus - minus)/2 ) 
                for plus, minus in zip(observables[0::2], observables[1::2])
            ] # zip goes through even and odd elements together
            gradient[start:end] = deriv
        
        # Last remaining n % n_qjobs components
        final_start = n-(n % half_qpus)
        final_results = gather([plus_minus for i in range(n % half_qpus) for plus_minus in shifted_i_circ(self.qjobs[i], self.qjobs[i+1], time_series, theta_now, final_start + i, shots=shots, nT=circuit.nT)])
        final_observables = [calc_observable(res, circuit.nE) for res in final_results]

        final_deriv = [ 
            np.dot( (cost_func.deriv(plus, y_true) - cost_func.deriv(minus, y_true))/2, (plus - minus)/2 ) 
            for plus, minus in zip(final_observables[0::2], final_observables[1::2])
        ]
    
        gradient[final_start:n] = final_deriv

        return np.array(gradient)

    def gradient_with_bias(self):
        """ """
        pass

    #################### INTERFACE METHODS ####################

    def update_gradient_method(self, new_gradient_method):
        f"""
        Function to change the gradient method to a new one. 

        Args:
            new_gradient_method (str): name of the new gradient method to set. Must be one of the supported ones: {self.supported_methods}.
        """
        logger.debug(f"Changing gradient method from {self.choice_method} to {new_gradient_method}")
        self.__init__(new_gradient_method)

    def __call__(self, *args, **kwds):
        return self.method(*args, **kwds)

#################### AUXILIARY METHODS ####################

def calc_observable(result, nE, observable = None) -> np.array:
    """
    Retrieves probabilities and calculates the chosen observable (only all Z's supported as of yet).

    Args:
        result (cunqa.result.Result class): property `result` of a qjob object
        nE (int): number of qubits of the Exchange register
        observable (None): not implemented yet
    """
    # TODO: improve this method to a flexible one once we have an observable calculation pipeline
    probs_dict = result.probabilities(per_qubit = True, partial = list(range(nE)))
    probs = np.concatenate(tuple([prob for prob in probs_dict.values()]))

    return np.array([prob_qubit[0]-prob_qubit[1] for prob_qubit in probs]) # ad hoc calculation of <Z> observable

    

def perturbed_i_circ(qjob: QJob, time_series: np.array, theta: np.array, index: int, diff: float, shots: int, nT: int) -> QJob:
    """
    Creates jobs with the QRNN circuit with 'theta' modified by 'diff' on coordinate 'index'.
    """
    theta_aux = np.copy(theta); theta_aux[index] += diff 
    logger.warning(f"Modified theta is {theta_aux}")

    param_dict = {}
    param_dict |= {var: x*np.pi for var, x     in zip(variables(f'x{nT}:{len(time_series)/nT}'), time_series)}
    param_dict |= {var: theta   for var, theta in zip(variables(f'theta:{len(theta_aux)}'),        theta_aux)}
    
    return qjob.upgrade_parameters(param_dict, shots=shots) # HARDCODED names for the params

def shifted_i_circ(qjob_minus: QJob, qjob_plus: QJob, time_series: np.array, theta: np.array, index: int, shots: int, nT: int)-> QJob:
    """
    Creates two jobs with the QRNN circuit with 'theta' modified by +pi/2 and -pi/2 on coordinate 'index'.
    """
    theta_plus  = np.copy(theta); theta_plus[index] += np.pi/2
    theta_minus = np.copy(theta); theta_minus[index] -= np.pi/2

    param_dict_plus = {}
    param_dict_plus |= (x_dict := {var: x*np.pi for var, x     in zip(variables(f'x{nT}:{len(time_series)/nT}'), time_series)})
    param_dict_plus |= {var: theta   for var, theta in zip(variables(f'theta:{len(theta_plus)}'),      theta_plus)}
    
    param_dict_minus = {}
    param_dict_minus |= x_dict
    param_dict_minus |= {var: theta   for var, theta in zip(variables(f'theta:{len(theta_minus)}'),    theta_minus)}

    qjob_minus.upgrade_parameters(param_dict_plus, shots=shots) # HARDCODED names for the params
    qjob_plus.upgrade_parameters(param_dict_minus, shots=shots) # HARDCODED names for the params

    return qjob_plus, qjob_minus

