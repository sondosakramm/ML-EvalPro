from abc import ABC, abstractmethod


class SummaryGenerator(ABC):
    """
    Abstract base class for summary generators.

    This class defines the structure for summary generators, requiring any subclass
    to implement the abstract method get_summary().

    Methods:
        get_summary(): Abstract method to be implemented by subclasses. It should
                      return a summary based on the specific module.
    """

    @abstractmethod
    def get_summary(self):
        """
        This method should be overridden by subclasses to provide a specific
        implementation for generating and returning a summary.

        Returns:
            str: The generated summary as a string based on the specific module.
        """
        pass
