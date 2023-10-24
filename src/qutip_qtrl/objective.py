class Objective:
    """
    A class for storing all
    information about an optimization objective.

    *Examples*
    >>> initial = qt.basis(2, 0)
    >>> target = qt.basis(2, 1)

    >>> sin = lambda t, p: p[0] * np.sin(t)
    >>> def d_sin(t, p, idx):
    >>>     if idx==0: return np.sin(t)
    >>>     if idx==1: return p[0] * np.cos(t)
    >>> H = [qt.sigmax(), [qt.sigmay(), sin, {'grad': d_sin}]]

    >>> obj = Objective(initial, H, target)

    Attributes
    ----------
    initial : :class:`qutip.Qobj`
        The initial state or operator to be transformed.
    H : callable, list or Qobj
        A specification of the time-depedent quantum object.
        See :class:`qutip.QobjEvo` for details and examples.
    target : :class:`qutip.Qobj`
        The target state or operator.
    """
    def __init__(self, initial, H, target):
        self.initial = initial
        self.H = H
        self.target = target

    def __getstate__(self):
        """
        Extract picklable information from the objective.
        Callable functions will be lost.
        """
        only_H = [self.H[0]] + [H[0] for H in self.H[1:]]
        return (self.initial, only_H, self.target)
