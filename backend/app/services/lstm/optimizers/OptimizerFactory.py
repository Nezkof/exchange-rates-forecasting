from app.services.lstm.optimizers.SGD import SGD
from app.services.lstm.optimizers.Adam import ADAM

class OptimizerFactory:
    _optimizers = {
        "ADAM": ADAM,
        "SGD": SGD,
    }

    @staticmethod
    def create_optimizer(optimizer_type, hidden_size, features_number, output_size, learning_rate):
        optimizer_class = OptimizerFactory._optimizers.get(optimizer_type.upper())
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        return optimizer_class(hidden_size, features_number, output_size, learning_rate)
