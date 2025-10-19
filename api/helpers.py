import typing

from src import predict


class ModelManager:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None
    
    def load_model(self):
        if self._tokenizer is None:
            self._tokenizer, self._model, self._device = predict.load_model('outputs/run')
        return self._tokenizer, self._model, self._device
    
    def predict(self, request: str) -> typing.List[dict]:
        tokenizer, model, device = self.load_model()
        return predict.predict_sample(request, tokenizer, model, device)


# Создаем единственный экземпляр менеджера
model_manager = ModelManager()
