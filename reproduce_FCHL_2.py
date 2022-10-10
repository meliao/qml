import argparse
import os
from typing import Dict, List, Tuple
import logging
from timeit import default_timer

import numpy as np
from scipy import io
import qml
from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
from qml.kernels import get_local_kernel, get_local_symmetric_kernel

FMT = "%(asctime)s:FCHL_regression_QM7: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'

def get_representations(coords: np.ndarray,
                        charges: np.ndarray,
                        n_atoms: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        coords (np.ndarray): Has shape (n_samples, max_n_atoms, 3)
        charges (np.ndarray): Has shape (n_samples, max_n_atoms)
        n_atoms (np.ndarray): Has shape (n_samples,)

    Returns:
        np.ndarray: Has shape (n_samples, max_n_atoms, 5, max_n_atoms)
    """
    out_lst = []
    for i in range(charges.shape[0]):
        n_atoms_i = n_atoms[i]

        x = generate_fchl_acsf(coords[i, :n_atoms_i], charges[i, :n_atoms_i], **KWARGS_ENERGY)
        out_lst.append(x)

    return np.array(out_lst)

def load_QM7(fp: str, n_train: int, n_test: int) -> Tuple[np.ndarray]:
    """_summary_

    Args:
        fp (str): _description_
        n_train (int): _description_
        n_test (int): _description_

    Returns:
        Tuple[np.ndarray]: _description_
    """
    data = io.loadmat(fp)

    #Atomization energies
    labels = data['T'].flatten()

    # Coordinates and charges
    coords = data['R'] / CONST_ANGSTROM_TO_BOHR_RADIUS
    charges = data['Z']

    # CV folds
    folds = data['P']

    train_idxes = folds[0:4].flatten()
    test_idxes = folds[4].flatten()

    # Split into train and test
    train_coords = coords[train_idxes]
    train_charges = charges[train_idxes]
    train_labels = labels[train_idxes]

    test_coords = coords[test_idxes]
    test_charges = charges[test_idxes]
    test_labels = labels[test_idxes]



    # Truncate
    train_coords = train_coords[:n_train]
    train_charges = train_charges[:n_train]
    train_labels = train_labels[:n_train]

    test_coords = test_coords[:n_test]
    test_charges = test_charges[:n_test]
    test_labels = test_labels[:n_test]

    return train_coords, train_charges, train_labels, test_coords, test_charges, test_labels


def loss_MAE(preds: np.ndarray, actual: np.ndarray) -> float:
    return np.mean(np.abs(preds.flatten() - actual.flatten()))


def write_result_to_file(fp: str, missing_str: str='', **trial) -> None:
    """Write a line to a tab-separated file saving the results of a single
        trial.
    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None
    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        header_line = "\t".join(header_lst) + "\n"
        with open(fp, 'w') as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, 'a') as f:
        f.write(trial_line)

CONST_ANGSTROM_TO_BOHR_RADIUS = 1.88973
KWARGS_ENERGY = {
    'nRs2': 22,
    'nRs3': 17,
    'eta2': 0.41,
    'eta3': 0.97,
    'three_body_weight': 45.83,
    'three_body_decay': 2.39,
    'two_body_decay': 2.39,
 }


def main(args: argparse.Namespace) -> None:

    #####################################################################
    ### LOAD DATA
    data = load_QM7(args.data_fp, args.n_train, args.n_test)
    
    train_coords = data[0]
    train_charges = data[1]
    train_labels = data[2]

    test_coords = data[0]
    test_charges = data[1]
    test_labels = data[2]
    

    logging.info("Train charges.shape: %s", train_charges.shape)
    logging.info("Train coords.shape: %s", train_coords.shape)


    train_n_atoms = np.sum(train_charges != 0, axis=1)
    test_n_atoms = np.sum(test_charges != 0, axis=1)


    #####################################################################
    ### CREATE TRAIN REPRESENTATIONS

    train_representations = get_representations(train_coords, train_charges, train_n_atoms)
    precompute_start = default_timer()
    test_representations = get_representations(test_coords, test_charges, test_n_atoms)
    precompute_time = default_timer() - precompute_start
    ####################################################################
    ### CREATE TRAIN DATA KERNEL

    sigmas = [1., 2., 4., 8., 16., 32.]
    reg_params = [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02]

    for i, sigma in enumerate(sigmas):
        logging.info("Looking at sigma=%f", sigma)

        kernel_mat = get_local_symmetric_kernel(train_representations, train_charges, sigma)
        kernel_start = default_timer()
        kernel_mat_test = get_local_kernel(train_representations, test_representations, train_charges, test_charges, sigma)
        kernel_time = default_timer() - kernel_start



        for reg_param in reg_params:
            logging.info("Looking at reg param = %f", reg_param)

            kernel_mat[np.diag_indices_from(kernel_mat)] += reg_param

            #####################################################################
            ### SOLVE FOR WEIGHTS
            logging.info("Solving for weights")
            alpha = cho_solve(kernel_mat, train_labels)

            train_preds = np.matmul(kernel_mat, alpha)
            train_MAE = loss_MAE(train_preds, train_labels)


            # logging.info("K train shape: %s", k_train.shape)
            # logging.info("Alpha shape: %s", alpha.shape)

            pred_time_start = default_timer()
            test_preds = np.matmul(kernel_mat_test.transpose(), alpha)
            pred_time = default_timer() - pred_time_start



            logging.info("Preds shape: %s", test_preds.shape)

            test_MAE = loss_MAE(test_preds, test_labels)
            logging.info("Train MAE (kcal/mol): %f and test MAE (kcal/mol): %f", train_MAE, test_MAE)

            experiment_dd = {
                'n_train': train_coords.shape[0],
                'n_test': test_coords.shape[1],
                'sigma': sigma,
                'reg_param': reg_param,
                'test_MAE': test_MAE,
                'train_MAE': train_MAE,
                'precompute_time': precompute_time,
                'kernel_time': kernel_time,
                'pred_time': pred_time,
                'n_sigmas_tested': len(sigmas)
            }

            write_result_to_file(args.results_fp, **experiment_dd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_fp', default='data/qm7/qm7.mat')
    parser.add_argument('-results_fp')
    parser.add_argument('-n_train', type=int, default=100)
    parser.add_argument('-n_test', type=int, default=1000)

    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)