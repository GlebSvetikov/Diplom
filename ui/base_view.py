from abc import ABC, abstractmethod

class BaseView(ABC):
    @abstractmethod
    def setup_ui(self):
        pass
