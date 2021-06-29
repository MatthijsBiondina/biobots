from abc import ABC, abstractmethod


class AbstractStoppingCondition(ABC):
    """
    This class gives the details for how stopping conditions must be implemented
    """
    @property
    @abstractmethod
    def name(self):
        """
        Give the stopping condition a name for user feedback

        :return: condition name
        """
        pass

    @abstractmethod
    def has_stopping_condition_been_met(self, t):
        pass

    def check_stopping_condition(self, t):
        stopped = self.has_stopping_condition_been_met(t)

        if stopped:
            print(f"Simulation stopped by {self.name}")
