"""
This module contains functions that implement the GOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import qutip as qt
from qutip import Qobj, QobjEvo

import numpy as NP  # might be overwritten by jax.numpy
import scipy as sp

#from result import Result


def enable_jnp():
    """
    Switch from NumPy(default) to JAX NumPy.
    """
    global jax, qutip_jax, NP
    try:
        import jax
        import qutip_jax
        import jax.numpy as NP
    except ImportError:
        raise ImportError(
            "JAX not available, make sure jax and qutip_jax are installed."
        )


class GOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    """
    # calculated during optimization
    g = None  # normalized overlap of U and target
    U = None  # current evolution operator
    dU = None  # derivative of U wrt control parameters
    infid = None  # projective SU distance (infidelity)

    def __init__(self, objective, evo_time, para_counts, var_t, **integrator_options):

        # extract drift and control Hamiltonians from the objective
        self.Hd = objective.H[0]
        self.Hc_lst = [H[0] for H in objective.H[1:]]

        # extract initial and target states from the objective
        self.initial = objective.initial
        self.target = objective.target

        # extract control functions and gradients from the objective
        self.controls = [H[1] for H in objective.H[1:]] 
        self.grads = [H[2]["grad"] for H in objective.H[1:]]

        self.evo_time = evo_time  # [T_i, T_f]
        self.para_counts = para_counts  # num of params for each control function
        self.var_t = var_t  # Boolean, whether time is a parameter
        self.integrator_options = integrator_options  # for SESolver

        # inferred attributes
        self.norm = 1 / self.target.shape[0] if self.target.isoper else 1
        self.sys_size = self.Hd.shape[0]
        self.tot_n_para = sum(self.para_counts)  # incl. time if var_t==True

        # jit compile functions if using JAX
        self.useJAX = integrator_options.get("method", "") == "diffrax"

        if self.useJAX:
            self.solve_EOM = self.solve_EOM_jax
            self.comp_infidelity = jax.jit(self.comp_infid)
            self.initial.to("jax")
            self.target.to("jax")
            # TODO: jit comp_grad
        else:
            self.solve_EOM = self.solve_EOM_numpy
            self.comp_infidelity = self.comp_infid

        # Scale the system Hamiltonian and initial state
        # for coupled system (U, dU)
        self.H = self.prepare_H()
        self.dH = self.prepare_dH()
        self.psi0 = self.prepare_psi0()

        # initialize the solver
        # TODO: why does QobjEvo require init_params?
        init_params = NP.ones(self.tot_n_para)
        self.solver = qt.SESolver(
            QobjEvo(self.H + self.dH, {"p": init_params}),
            options=self.integrator_options
        )

    def prepare_psi0(self):
        """
        inital state for coupled system (U, dU):
        [[  H, 0, 0, ...], [[  U],
         [d1H, H, 0, ...],  [d1U],
         [d2H, 0, H, ...],  [d2U],
         [...,         ]]   [...]]
        """
        scale = sp.sparse.csr_matrix(
            ([1], ([0], [0])), shape=(1 + self.tot_n_para, 1)
        )
        psi0 = Qobj(scale) & self.initial
        return psi0 if self.useJAX == False else psi0.to("jax")

    def prepare_H(self):
        """
        Combines the scaled Hamiltonian with associated control pulses
        """
        def coeff(control, lower, upper):
            # helper function to fix arguments in loop
            if self.useJAX == False:
                return lambda t, p: control(t, p[lower:upper])
            else:  # coeff must be jit compiled to be recognized by qutip_jax
                return jax.jit(lambda t, p: control(t, p[lower:upper]))

        diag = qt.qeye(1 + self.tot_n_para)
        H = [diag & self.Hd]
        idx = 0

        # H = [Hd, [H0, c0(t)], ...]

        for control, M, Hc in zip(self.controls, self.para_counts, self.Hc_lst):
            hc = diag & Hc if self.useJAX == False else (diag & Hc).to("jax")
            H.append([hc, coeff(control, idx, idx + M)])
            idx += M
        return H  # list to construct QobjEvo

    def prepare_dH(self):
        """
        Combines the scaled Hamiltonian with the derivative of the associated pulses
        """
        def coeff(grad, lower, upper, idx):
            # helper function to fix arguments in loop
            if self.useJAX == False:
                return lambda t, p: grad(t, p[lower:upper], idx)
            else:  # coeff must be jit compiled to be recognized by qutip_jax
                return jax.jit(lambda t, p: grad(t, p[lower:upper], idx))

        csr_shape = (1 + self.tot_n_para,  1 + self.tot_n_para)
        dH = []
        idx = 0

        # dH = [[H1', dc1'(t)], [H1", dc1"(t)], ... , [H2', dc2'(t)], ...]

        for grad, M, Hc in zip(self.grads, self.para_counts, self.Hc_lst):
            for grad_idx in range(M + int(self.var_t)):
                i = 1 + idx + grad_idx if grad_idx < M else self.tot_n_para
                csr = sp.sparse.csr_matrix(([1], ([i], [0])), csr_shape)
                hc = Qobj(csr) & Hc if self.useJAX == False else\
                    (Qobj(csr) & Hc).to("jax")
                dH.append([hc, coeff(grad, idx, idx + M, grad_idx)])
            idx += M
        return dH  # list to construct QobjEvo

    def solve_EOM_numpy(self, evo_time, params):
        """
        Calculates U, and dU i.e. the derivative of the evolution operator U
        wrt the control parameters by solving the Schrodinger operator equation
        """
        res = self.solver.run(self.psi0, evo_time, args={'p': params})

        U = res.final_state[:self.sys_size, :self.sys_size]
        dU = res.final_state[self.sys_size:, :self.sys_size]
        return U, dU

    def solve_EOM_jax(self, evo_time, params):
        """
        Calculates U, and dU i.e. the derivative of the evolution operator U
        wrt the control parameters by solving the Schrodinger operator equation
        """
        res = self.solver.run(self.psi0, evo_time, args={'p': params})

        jax_res = res.final_state.data._jxa
        U = jax.lax.slice(jax_res,
                          (0, 0),
                          (self.sys_size, self.sys_size))
        dU = jax.lax.slice(jax_res,
                           (self.sys_size, 0),
                           (jax_res.shape[0], self.sys_size))
        return U, dU

    def comp_infid(self, params):
        """
        projective SU distance (infidelity) to be minimized
        store intermediate results for gradient calculation
        returns the infidelity, the normalized overlap,
        the current unitary and its gradient for later use
        """
        # adjust integration time interval, if time is parameter
        evo_time = [self.evo_time[0], self.evo_time[-1]
                    if self.var_t == False else params[-1]]

        U, dU = self.solve_EOM(evo_time, params)

        U = Qobj(U, dims=self.target.dims)
        g = self.norm * U.overlap(self.target)
        infid = 1 - NP.abs(g)

        return infid, g, U, dU  # f_PSU

    def comp_grad(self):
        """
        Calculates the gradient of the fidelity error function
        wrt control parameters by solving the Schrodinger operator equation
        according to GOAT algorithm: arXiv:1507.04261
        """
        dU, g = self.dU, self.g  # both calculated before

        trc = []  # collect for each parameter
        for i in range(self.tot_n_para):
            idx = i * self.sys_size  # row index for parameter set i
            if self.useJAX:
                slice = jax.lax.slice(
                    dU, (idx, 0), (idx + self.sys_size, self.sys_size))
            else:
                slice = dU[idx: idx + self.sys_size, :]
            du = Qobj(slice)
            trc.append(self.target.overlap(du))

        # NOTE: gradient will be zero at local maximum
        fact = (NP.conj(g) / NP.abs(g)) * self.norm if g != 0 else 0
        grad = -(fact * NP.array(trc)).real  # -Re(... * Tr(...))
        return grad


class Multi_GOAT:
    """
    Composite class for multiple GOAT instances
    to optimize multiple objectives simultaneously
    """

    def __init__(self, objectives, evo_time, para_counts, var_t, **integrator_options):
        self.goats = [GOAT(obj, evo_time, para_counts, var_t, ** integrator_options)
                      for obj in objectives]
        self.mean_infid = None

    def comp_grad(self, params):
        grads = 0
        for g in self.goats:
            grads += g.comp_grad()
        return grads

    def goal_fun(self, params):
        i = 0
        for goat in self.goats: # TODO: parallelize
            goat.infid, goat.g, goat.U, goat.dU = goat.comp_infidelity(params)
            if goat.infid < 0:
                print(
                    "WARNING: infidelity < 0 -> inaccurate integration,\
                    try reducing integrator tolerance (atol, rtol)"
                )
                goat.infid = 1.  # TODO: should penalize result?
            i += goat.infid
        self.mean_infid = NP.mean(i)
        return self.mean_infid

    def get_parameterized_controls(self):
        """
        Extract the parameterized control functions from a single GOAT instance
        Handy for storing and plotting the optimized controls
        """
        return [H[1] for H in self.goats[0].H[1:]]


def optimize_pulses(
        objectives,
        pulse_options,
        tlist,
        kwargs={}):
    """
    Optimize a pulse sequence to implement a given target unitary.

    Parameters
    ----------
    objectives : list of :class:`qutip.Qobj`
        List of objectives to be implemented.
    pulse_options : dict of dict
        Dictionary of options for each pulse.
        guess : list of floats
            Initial guess for the pulse parameters.
        bounds : list of pairs of floats
            [(lower, upper), ...]
            Bounds for the pulse parameters.
    tlist : array_like
        List of times for the calculataion of final pulse sequence.
        During integration only the first and last time are used.
    kwargs : dict of dict
        Dictionary keys are "optimizer", "minimizer" and "integrator".

        The "optimizer" dictionary contains keyword arguments for the optimizer:
            niter : integer, optional
                The number of basin-hopping iterations. There will be a total of
                ``niter + 1`` runs of the local minimizer.
            T : float, optional
                The "temperature" parameter for the acceptance or rejection criterion.
                Higher "temperatures" mean that larger jumps in function value will be
                accepted.  For best results `T` should be comparable to the
                separation (in function value) between local minima.
            stepsize : float, optional
                Maximum step size for use in the random displacement.
            take_step : callable ``take_step(x)``, optional
                Replace the default step-taking routine with this routine. The default
                step-taking routine is a random displacement of the coordinates, but
                other step-taking algorithms may be better for some systems.
                `take_step` can optionally have the attribute ``take_step.stepsize``.
                If this attribute exists, then `basinhopping` will adjust
                ``take_step.stepsize`` in order to try to optimize the global minimum
                search.
            accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
                Define a test which will be used to judge whether to accept the
                step. This will be used in addition to the Metropolis test based on
                "temperature" `T`. The acceptable return values are True,
                False, or ``"force accept"``. If any of the tests return False
                then the step is rejected. If the latter, then this will override any
                other tests in order to accept the step. This can be used, for example,
                to forcefully escape from a local minimum that `basinhopping` is
                trapped in.
            callback : callable, ``callback(x, f, accept)``, optional
                A callback function which will be called for all minima found. ``x``
                and ``f`` are the coordinates and function value of the trial minimum,
                and ``accept`` is whether that minimum was accepted. This can
                be used, for example, to save the lowest N minima found. Also,
                `callback` can be used to specify a user defined stop criterion by
                optionally returning True to stop the `basinhopping` routine.
            interval : integer, optional
                interval for how often to update the `stepsize`
            disp : bool, optional
                Set to True to print status messages
            niter_success : integer, optional
                Stop the run if the global minimum candidate remains the same for this
                number of iterations.
            seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional

                If `seed` is None (or `np.random`), the `numpy.random.RandomState`
                singleton is used.
                If `seed` is an int, a new ``RandomState`` instance is used,
                seeded with `seed`.
                If `seed` is already a ``Generator`` or ``RandomState`` instance then
                that instance is used.
                Specify `seed` for repeatable minimizations. The random numbers
                generated with this seed only affect the default Metropolis
                `accept_test` and the default `take_step`. If you supply your own
                `take_step` and `accept_test`, and these functions use random
                number generation, then those functions are responsible for the state
                of their random number generator.
            target_accept_rate : float, optional
                The target acceptance rate that is used to adjust the `stepsize`.
                If the current acceptance rate is greater than the target,
                then the `stepsize` is increased. Otherwise, it is decreased.
                Range is (0, 1). Default is 0.5.
            See scipy.optimize.basinhopping for more details.

        The "minimizer" dictionary contains keyword arguments for the minimizer:
            method : str or callable, optional
                Type of solver.  Should be one of
                    - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
                    - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
                    - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
                    - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
                    - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
                    - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
                    - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
                    - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
                    - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
                    - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
                    - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
                    - custom - a callable object, see below for description.
                If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
                depending on whether or not the problem has constraints or bounds.
            bounds : sequence or `Bounds`, optional
                Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
                trust-constr, and COBYLA methods. There are two ways to specify the
                bounds:
                    1. Instance of `Bounds` class.
                    2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                    is used to specify no bound.
            constraints : {Constraint, dict} or List of {Constraint, dict}, optional
                Constraints definition. Only for COBYLA, SLSQP and trust-constr.

                Constraints for 'trust-constr' are defined as a single object or a
                list of objects specifying constraints to the optimization problem.
                Available constraints are:
                    - `LinearConstraint`
                    - `NonlinearConstraint`
                Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
                Each dictionary with fields:
                    type : str
                        Constraint type: 'eq' for equality, 'ineq' for inequality.
                    fun : callable
                        The function defining the constraint.
                    jac : callable, optional
                        The Jacobian of `fun` (only for SLSQP).
                    args : sequence, optional
                        Extra arguments to be passed to the function and Jacobian.
                Equality constraint means that the constraint function result is to
                be zero whereas inequality means that it is to be non-negative.
                Note that COBYLA only supports inequality constraints.
            tol : float, optional
                Tolerance for termination. When `tol` is specified, the selected
                minimization algorithm sets some relevant solver-specific tolerance(s)
                equal to `tol`. For detailed control, use solver-specific
                options.
            options : dict, optional
                A dictionary of solver options. All methods except `TNC` accept the
                following generic options:
                    maxiter : int
                        Maximum number of iterations to perform. Depending on the
                        method each iteration may use several function evaluations.
                        For `TNC` use `maxfun` instead of `maxiter`.
                    disp : bool
                        Set to True to print convergence messages.
                For method-specific options, see :func:`show_options()`.
            See scipy.optimize.minimize for more details.

        The "integrator" dictionary contains keyword arguments for the integrator options:
                - progress_bar : str {'text', 'enhanced', 'tqdm', ''}
                How to present the integrator progress.
                'tqdm' uses the python module of the same name and raise an error
                if not installed. Empty string or False will disable the bar.
                - progress_kwargs : dict
                kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
                - method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
                Which differential equation integration method to use.
                - atol, rtol : float
                Absolute and relative tolerance of the ODE integrator.
                - nsteps : int
                Maximum number of (internally defined) steps allowed in one ``tlist``
                step.
                - max_step : float, 0
                Maximum lenght of one internal step. When using pulses, it should be
                less than half the width of the thinnest pulse.

                Other options could be supported depending on the integration method,
                see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    """
    optimizer_kwargs = kwargs.get("optimizer", {})
    minimizer_kwargs = kwargs.get("minimizer", {})
    integrator_kwargs = kwargs.get("integrator", {})

    # integrator must not normalize output
    integrator_kwargs["normalize_output"] = False

    # extract initial and boundary values
    x0, bounds, para_counts = [], [], []
    for key in pulse_options.keys():
        x0.append(pulse_options[key].get("guess"))
        bounds.append(pulse_options[key].get("bounds"))

    optimizer_kwargs["x0"] = NP.concatenate(x0)
    minimizer_kwargs["bounds"] = NP.concatenate(bounds)

    # number of parameters for each single control function
    para_counts = [len(x) for x in x0]

    # optimize time, when "time" is in pulse_options
    var_t = True if pulse_options.get("time", False) else False
    evo_time = NP.array([tlist[0], tlist[-1]])
    
    # import qutip_jax, jax.numpy as NP if needed
    use_jax = integrator_kwargs.get("method", "") == "diffrax"
    selection = "jax" if use_jax else None
    if use_jax: enable_jnp()

    # initialize the Multi_GOAT instance
    with qt.CoreOptions(default_dtype=selection):
        goats = Multi_GOAT(objectives, evo_time, para_counts,
                           var_t, **integrator_kwargs)

    ## define the result Krotov style
    #result = Result(tlist, objectives)
    #result.start_time()

    min_res = sp.optimize.basinhopping(
        func=goats.goal_fun,
        minimizer_kwargs={
            'jac': goats.comp_grad,
            'callback': lambda xk: print(
                "Minimizer step, infidelity: ", goats.mean_infid),
            **minimizer_kwargs
        },
        callback=lambda xk, f, accept: print(
            "took $%.2f seconds" 
            #%result.time_iter()
            ),
        **optimizer_kwargs
    )

    #result.end_time()
    #result.generate_optimized_controls(
    #    optimizer_kwargs["x0"], min_res.x, goats.get_parameterized_controls()
    #)
    #result.final_result(
    #    U_final=[g.U for g in goats.goats],
    #    min_res=min_res,
    #)

    return min_res
