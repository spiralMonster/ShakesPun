import math
from tensorflow.keras.callbacks import LearningRateScheduler

class CosineAnnealingScheduler(LearningRateScheduler):
    
    def __init__(self,initial_learning_rate,total_epochs):
        self.total_epochs=total_epochs
        self.initial_lr=initial_learning_rate
        super().__init__(self.scheduler_function,verbose=0)
        
    def scheduler_function(self,epoch,lr):
        return 0.5*self.initial_lr*(1+math.cos(math.pi*(epoch/self.total_epochs)))
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'initial_lr':self.initial_lr,
                'total_epochs':self.total_epochs
            }
        )
        return config
       
        


    