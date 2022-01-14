import numpy as np
class OptimizadorAdam():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def actualizar(self, t, w, b, dw, db):
        ## dw, db son del minibatch actual
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        ## bias correccion
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        ## actualizar pesos y biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w, b



def funcionDePerdida(m):
    return m**2-2*m+1
## take derivative
def funcionGradiente(m):
    return 2*m-2
def checarConvergencia(w0, w1):
    return (w0 == w1)

w_0 = 0
b_0 = 0
adam = OptimizadorAdam()
t = 1 
converged = False

while not converged:
    dw = funcionGradiente(w_0)
    db = funcionGradiente(b_0)
    w_0_old = w_0
    w_0, b_0 = adam.actualizar(t,w=w_0, b=b_0, dw=dw, db=db)
    if checarConvergencia(w_0, w_0_old):
        print('convergido despues de: '+str(t)+' iteraciones')
        break
    else:
        print('iteracion '+str(t)+': peso='+str(w_0))
        t+=1

