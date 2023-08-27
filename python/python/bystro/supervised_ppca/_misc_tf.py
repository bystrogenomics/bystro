"""
This contains various miscellaneous but commonly used methods in tensorflow,
namely preventing TF from taking over your GPU, and picking optimizers.

Methods
-------
def copy_model(model_origin,model_target):
    Copies a particular sequential model to a new model

return_optimizer_tf(trainingMethod,learningRate,options=None)
    Creates a Keras optimizer with specific parameters

return_optimizer_adaptive_tf(learningRate,options)
    Creates a Keras optimizer with specific parameters

limitGPU(gpuMem)
    Limits the GPU memory to a certain amount
"""
from tensorflow import keras
import tensorflow as tf


def copy_model(model_origin, model_target):
    """
    Copies a particular sequential model to a new model

    Parameters
    ----------
    model_origin : tensorflow model
        The model with learned weights

    model_target : tensorflow model
        Model with unlearned weights

    Returns
    -------
    model_target : tensorflow model
        The model with identical weights to model_origin
    """
    for l_orig, l_targ in zip(model_origin.layers, model_target.layers):
        l_targ.set_weights(l_orig.get_weights())
    return model_target


def limit_gpu(gpuMem):
    """
    Limits the GPU memory to a certain amount

    Parameters
    ----------
    gpuMem : int
        MB of memory to allocate
    """
    gpuMem = int(gpuMem)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpuMem
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                " Physical GPUs, ",
                len(logical_gpus),
                " Logical GPUs",
            )
        except RuntimeError as e:
            print(e)


def return_optimizer_tf(trainingMethod, learningRate):
    """
    Creates a Keras optimizer with specific parameters

    Parameters
    ----------
    trainingMethod : str \in {'Nadam','Adam', 'SGD'}
        The SGD method

    learningRate : float
        The learning rate of optimization

    options : dict
        Misc options, currently unused

    Returns
    -------
    optimizer : keras optimizer
    """
    if trainingMethod == "Nadam":
        optimizer = keras.optimizers.Nadam(learning_rate=learningRate)
    elif trainingMethod == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    elif trainingMethod == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    else:
        raise ValueError("Unrecognized learning strategy %s", trainingMethod)
    return optimizer


def return_optimizer_adaptive_tf(learningRate, options):
    """
    Creates a Keras optimizer with specific parameters

    Parameters
    ----------
    learningRate : float
        The learning rate of optimization

    options : dict
        Misc scheduling options

    Returns
    -------
    optimizer : keras optimizer
    """
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learningRate,
        decay_steps=options["steps"],
        decay_rate=options["rate"],
        staircase=options["staircase"],
    )
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    return optimizer


def subset_square_matrix_tf(Sigma, idxs):
    """
    Obtains a subset of a square matrix identical in both directions

    Parameters
    ----------
    Sigma : tf.array,shape=(p,p)
        The matrix to subset

    idxs : np.array,shape=(p,)
        Boolean indexes to select

    Returns
    -------
    SSub : tf.array,shape=(sum(idxs),sum(idxs))
        The subset matrix
    """
    SSub1 = tf.boolean_mask(Sigma, idxs, axis=1)
    SSub = tf.boolean_mask(SSub1, idxs, axis=0)
    return SSub


def subset_matrix_tf(M, idxs1, idxs2):
    """
    Obtains a subset of a general matrix

    Parameters
    ----------
    M : tf.array,shape=(p,q)
        The matrix to subset

    idxs1 : np.array,shape=(p,)
        Boolean indexes to select the first axis

    idxs2 : np.array,shape=(p,)
        Boolean indexes to select the second axis

    Returns
    -------
    SSub : tf.array,shape=(sum(idxs),sum(idxs))
        The subset matrix
    """
    SSub1 = tf.boolean_mask(M, idxs1, axis=0)
    SSub = tf.boolean_mask(SSub1, idxs2, axis=1)
    return SSub
