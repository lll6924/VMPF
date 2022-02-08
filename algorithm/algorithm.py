from abc import abstractmethod

class Algorithm:
    @abstractmethod
    def loss(self):
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_variables(self):
        raise NotImplementedError()