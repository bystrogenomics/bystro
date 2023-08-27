"""


"""
import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tqdm import trange
from ..utils._misc_tf import limitGPU, return_optimizer_tf
from ..utils._losses_tf import loss_offdiag_Lq_tf
from ..utils._batcher_np import simple_batcher
from ..utils._misc import fill_dict
from ._base_precision import BasePrecision


class PrecisionL1PenalizedTF(BasePrecision):
    def __init__(self, lambda_n=1.0, training_options={}):
        super(PrecisionL1PenalizedTF, self).__init__()
        self.lambda_n = float(lambda_n)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, X):
        # https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/High-dimensional-covariance-estimation-by-minimizing-%e2%84%931-penalized-log-determinant/10.1214/11-EJS631.full

        N, p = X.shape
        self.n_samples, self.p = X.shape
        td = self.training_options

        limitGPU(td["gpu_memory"])
        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        if td["batch_size"] is None:
            empirical_covariance = 1 / N * np.dot(X.T, X).astype(np.float32)

        cholesky_ = tf.Variable(np.eye(p).astype(np.float32))
        trainable_variables = [cholesky_]

        self._initialize_saved_losses()

        for i in trange(td["n_iterations"]):
            if td["batch_size"] is not None:
                Xb = simple_batcher(N, td["batch_size"])
                empirical_covariance = (
                    1 / td["batch_size"] * np.dot(Xb.T, Xb).astype(np.float32)
                )

            with tf.GradientTape() as tape:
                precision_ = tf.matmul(cholesky_, tf.transpose(cholesky_))
                loss_recon = tf.reduce_mean(empirical_covariance * precision_)
                loss_det = tf.linalg.logdet(precision_)
                loss_off = loss_offdiag_Lq_tf(precision_, q=1)

                loss = loss_recon - loss_det + self.lambda_n * loss_off

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, loss, loss_recon, loss_det, loss_off)

        self.precision = precision_.numpy()
        self.covariance = la.inv(self.precision)

    def _initialize_saved_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training

        Sets
        ----
        losses_objective : np.array(n_iterations)
            The total loss of our objective at each iteration

        losses_recon : np.array(n_iterations)
            The trace inner product

        losses_log_det : np.array(n_iterations)
            The loss corresponding to the log det of precision

        losses_off_diag : np.array(n_iterations)
            The off diagonal sparseness penalty
        """
        n_iterations = self.training_options["n_iterations"]
        self.losses_objective = np.zeros(n_iterations)
        self.losses_recon = np.zeros(n_iterations)
        self.losses_log_det = np.zeros(n_iterations)
        self.losses_off_diag = np.zeros(n_iterations)

    def _save_losses(self, iteration, loss, loss_recon, loss_det, loss_off):
        """
        This saves the losses during training. See _initialize_saved_losses
        for descriptions
        """
        self.losses_objective[iteration] = loss.numpy()
        self.losses_recon[iteration] = loss_recon.numpy()
        self.losses_log_det[iteration] = loss_det.numpy()
        self.losses_off_diag[iteration] = loss_off.numpy()

    def _fill_training_options(self, training_options):
        """
        This sets the default parameters for stochastic gradient descent,
        our inference strategy for the model.

        Parameters
        ----------
        training_options : dict
            The original options set by the user passed as a dictionary

        Options
        -------
        n_iterations : int, default=3000
            Number of iterations to train using stochastic gradient descent

        learning_rate : float, default=1e-4
            Learning rate of gradient descent

        method : string {'Nadam'}, default='Nadam'
            The learning algorithm

        batch_size : int, default=None
            The number of observations to use at each iteration. If none
            corresponds to batch learning

        gpu_memory : int, default=1024
            The amount of memory you wish to use during training
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 1e-3,
            "gpu_memory": 1024,
            "method": "Nadam",
            "batch_size": None,
        }
        return fill_dict(training_options, default_options)
