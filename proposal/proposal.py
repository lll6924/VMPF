from abc import abstractmethod

class GaussianProposal:
    @abstractmethod
    def LocAndScale(self, t, given,y_t):
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def diagnal(self):
        raise NotImplementedError()