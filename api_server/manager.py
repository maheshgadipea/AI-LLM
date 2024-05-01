class ModelTemplate():
    def __init__(self) -> None:
        self.model_obj = None
        
    
    def prediction_fn(self,prompt):
        return f"Your prompt is :{prompt}"
