"""
File: adam_learner.py
Description: functionality for Adam optimisation of a GP model
"""

from jax import jit, value_and_grad, random
import jax.numpy as jnp
from numpy import array2string
import optax

from typing import Callable

# fixed settings for learning rate schedule
DECAY_RATE = 0.99
STAIRCASE = False
TRANS_BEGIN = 0.75
END_LR_FACTOR = 25.


class AdamLearner:
    r"""
    Class for handling training of a GP model using Adam
    """

    def __init__(
        self,
        obj_fn: Callable,  # objective function to be minimised
        opt_options: dict,  # specifies details of the optimisation routine
        generate_params: Callable,  # generates model parameters given a PRNG key
        base_seed: int,  # base PRNG seed value
    ):

        # must specify number of training steps and learning rate
        for key in ["n_steps", "lr"]:
            assert key in opt_options, f"Must specify {key} in opt_options dict"

        self.init_adam_optimiser(obj_fn, opt_options["n_steps"], opt_options["lr"])

        self.generate_params = generate_params
        self.base_seed = base_seed

        # initialise the model parameters and optimiser state
        self.initialise_params_and_state()

        # initialise to very large value
        self.min_obj_value = 1e10

        # use softplus as default transform on noise parameters
        if "noise_transform" in opt_options:
            self.noise_trans = opt_options["noise_transform"]
        else:
            self.noise_trans = softplus

        # theta refers to any unknown PDE parameters
        self.theta_exists = "theta" in self.params
        if "theta_transform" in opt_options:
            self.theta_trans = opt_options["theta_transform"]
        else:
            self.theta_trans = lambda x, pass_: x

    def train(self, n_steps: int, print_progress: bool = True, n_print_steps: int = 25):
        r"""
        Trains GP model for specified number of optimisation steps and saves results to self.params_best

        Args:
            n_steps: number of steps to train for
            print_progress: whether to print progress to console during training
            n_print_steps: the number of progress steps to print to console, if print_progress=True
        """

        # number of training steps between each print of training progress
        print_step = max(1, n_steps / n_print_steps)

        # lists to record training progress
        obj_list = []
        theta_list = []

        # lists to record training results 10 steps behind current step
        params_minus_10 = []
        obj_minus_10 = []

        # train for n_steps
        for j in range(n_steps):

            # perform training for one step
            updated_params, updated_opt_state, new_obj = self.update_one_step(
                self.params, self.opt_state
            )

            # keep track of results from 10 steps ago, for use if an nan is encountered
            if len(params_minus_10) < 10:
                params_minus_10 += [updated_params]
                obj_minus_10 += [new_obj]
            else:
                params_minus_10 = params_minus_10[1:] + [updated_params]
                obj_minus_10 = obj_minus_10[1:] + [new_obj]

            # Catch any nan values
            if jnp.isnan(new_obj):
                print(f"WARNING: nan value encountered after {j} training steps")
                # reset parameters and objective to 10 steps before
                self.params = params_minus_10[0]
                obj = obj_minus_10[0]
                # end training
                break
            else:
                obj = new_obj
                self.params = updated_params
                self.opt_state = updated_opt_state

            if (j // print_step == j / print_step) or (j == n_steps - 1):

                # record values of obj function / PDE parameter theta
                obj_list.append(obj)
                if self.theta_exists:
                    theta_transformed = self.theta_trans(self.params["theta"])
                    theta_list.append(theta_transformed)

                # print training progress to console
                if print_progress and (j < (n_steps - 1)):
                    self.print_progress(j, obj)

        print(f"Final training results: ")
        self.print_progress(j, obj)
        print("\n")

        # save results if the value of the obj fn is lower than all earlier optimisations
        if obj < self.min_obj_value:
            self.params_best = self.params
            self.opt_state_best = self.opt_state
            self.min_obj_value = obj
            self.obj_list = obj_list
            self.theta_list = theta_list

    def train_with_restarts(
        self,
        n_steps: int,
        n_restarts: int,
        print_progress: bool = True,
        n_print_steps: bool = 25,
    ):
        r"""
        Trains GP model under n_restarts random initialisations

        Calls the train() method n_restarts times, where at each restart the kernel parameters are randomly
        regenerated and the optimiser state is reset. The results which yield lowest value of the objective
        function across the different restarts are saved
        """
        for i in range(n_restarts):
            print(f"Restart {i}: beginning training for {n_steps} steps")
            # reinitialise model parameters and optimisation steps
            self.initialise_params_and_state()
            # train for n_steps optimisation steps
            self.train(n_steps, print_progress, n_print_steps)

    def print_progress(self, j: int, obj_val: float):
        r"""
        Prints training progress to console
        """

        # format noise values into neater format for printing
        noise_std_str = array2string(
            self.noise_trans(self.params["noise_std"]),
            separator=",",
            formatter={"float_kind": lambda x: "%.2e" % x},
        )

        print_str = f"({j}): {obj_val:.4f}, noise_std_vals = {noise_std_str}"

        # if running inverse problem, append inferred value of theta to print_str
        if self.theta_exists:
            theta_transformed = self.theta_trans(self.params["theta"])
            print_str += f", theta = {theta_transformed}"

        # print to console
        print(print_str)

    def initialise_params_and_state(self):
        r"""
        Initialse parameters of GP model and state of the Adam optimiser
        """

        # generate new seed for generating params and replace old base_seed
        rng_seed, self.base_seed = random.split(self.base_seed, 2)

        self.params = self.generate_params(rng_seed)
        self.opt_state = self.tx.init(self.params)

    def init_adam_optimiser(self, obj_fn: Callable, n_steps: int, lr: float):
        r"""
        Initialises update_one_step method for training GP model using Adam

        Args:
            obj_fn: function to be minimised
            n_steps: number of steps to perform training for
            lr: initial learning rate of optimiser
        """

        # exponentially decaying learning rate schedule
        self.lr_schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=1,#n_steps // 2,
            decay_rate=DECAY_RATE,
            transition_begin=int(n_steps * TRANS_BEGIN),
            staircase=STAIRCASE,
            end_value=lr/END_LR_FACTOR
        )

        # initialise optimiser with specified lr schedule
        self.tx = optax.adam(self.lr_schedule)

        # function to evaluate the value and gradient of the objective function
        self.grad_obj_fn = value_and_grad(obj_fn)

        @jit
        def update_one_step(params: dict, opt_state):
            r"""
            Performs one step update of params using Adam optimiser
            """

            # evaluate value of objective fn and gradient wrt params
            obj, grads = self.grad_obj_fn(params)

            # update the optimiser state, and create an update to the params
            updates, opt_state = self.tx.update(grads, opt_state)

            # update the parameters
            params = optax.apply_updates(params, updates)

            return params, opt_state, obj

        self.update_one_step = update_one_step
