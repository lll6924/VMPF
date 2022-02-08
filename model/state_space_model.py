from abc import abstractmethod

class StateSpaceModel:
    @abstractmethod
    def logf(self, time, x_t, given=None):
        raise NotImplementedError()

    @abstractmethod
    def logg(self,y_t, x_t):
        raise NotImplementedError()