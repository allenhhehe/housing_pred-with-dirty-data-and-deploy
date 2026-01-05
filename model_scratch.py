import numpy as np
class linearRegressionGD:
    def __init__(self,lr:float=0.05,epochs:int=2000):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=0.0
        self.cost_history=[]

    def perdict(self,X:np.ndarray)->np.ndarray:
        return X @ self.w + self.b
    
    def fit(self,X:np.ndarray,y:np.ndarray):
        m,n=X.shape
        self.w=np.zeros(n,dtype=float)
        self.b=0.0
        
        for i in range(self.epochs):
            y_hat=self.perdict(X)
            error=y_hat-y
            cost=error @ error/2*m
            self.cost_history.append(cost)

            dw=(X.T @ error )/m
            db=error.mean()

            self.w-=self.lr*dw
            self.b-=self*db
        return self

        