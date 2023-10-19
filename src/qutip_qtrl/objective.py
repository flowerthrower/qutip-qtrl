class Objective:
    def __init__(self, initial, H, target):
        self.initial = initial
        self.H = H
        self.target = target

    def __getstate__(self):
        """Return state values to be pickled."""
        only_H = [self.H[0]] + [H[0] for H in self.H[1:]]
        return (self.initial, only_H, self.target)
