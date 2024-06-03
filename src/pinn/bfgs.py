import logging   # nopep8
logging.getLogger('tensorflow').setLevel(logging.ERROR)   # nopep8
logging.getLogger('tensorflow_probability').setLevel(logging.ERROR)   # nopep8

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import keras


class BFGS:
    def __init__(self, loss, var_list, options={}) -> None:
        self.loss = loss
        self.var_list = []
        for v in var_list:
            if isinstance(v, keras.src.backend.common.variables.KerasVariable):
                self.var_list.append(v.value)
            else:
                self.var_list.append(v)
        self.options = options
        self.func = self._function_factory()

    def _function_factory(self):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.var_list)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = numpy.prod(shape)
            idx.append(tf.reshape(
                tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.var_list[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
            params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.var_list)
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = self.loss(*self.loss_arguments)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.var_list)
            grads = tf.dynamic_stitch(idx, grads)

            # store loss value so we can retrieve later
            tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []

        return f

    def _optimizer(self, initial_position):
        return tfp.optimizer.bfgs_minimize(self.func, initial_position, **self.options)

    def minimize(self, *argv):
        init_params = tf.dynamic_stitch(self.func.idx, self.var_list)
        self.loss_arguments = argv
        results = self._optimizer(init_params)
        self.func.assign_new_model_parameters(results.position)
        return results


class LBFGS(BFGS):

    def _optimizer(self, initial_position):
        return tfp.optimizer.lbfgs_minimize(self.func, initial_position, **self.options)
