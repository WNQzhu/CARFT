from accelerate import Accelerator
class T:
    def __init__(self):
        self.acc = Accelerator(gradient_accumulation_steps=2)
        
