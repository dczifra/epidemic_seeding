from jax import grad
import jax.numpy as jnp

import jax.numpy as np
import matplotlib.pyplot as plt

def tanh(x):  # Define a function
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0

class Model:
    def __init__(self, I, R=0.0):
        self.reinit(I,R)
    
    def reinit(self, I, R=0.0):
        assert(I+R <= 1.0)
        self.S = 1.0-I-R
        self.I = I
        self.R = R

    def step(self, beta, mu):
        S,I,R = self.S, self.I, self.R
        self.S -= beta*S*I
        self.I += -mu*I+beta*I*S
        self.R += mu*I

    def run(self, iternum, beta, mu):
        history = np.zeros((iternum, 3))
        for step in range(iternum):
            self.step(beta, mu)
            history = history.at[step].set(np.array([self.S, self.I, self.R]))

        return history

def I(ß):
    I0 = 0.01
    mu = 0.2
    iternum = 20

    S,I,R = 1.0-I0,I0,0.0
    for i in range(iternum):
        S,I,R = S-ß*I*S, I-mu*I+ß*I*S, R+mu
    
    return I


def plot(histories, labels, plot_SR = False):
    plt.figure(figsize=(12,8))
    for history,label in zip(histories, labels):
        plt.plot(history[:,1], label=label+"(I)")
        if plot_SR:
            plt.plot(history[:,0], label=label+"_S")
            plt.plot(history[:,2], label=label+"_R")
        plt.legend()

m = Model(0.01)
hist1 = m.run(500, 0.15, 0.1)
m.reinit(0.01)
hist2 = m.run(500, 0.2, 0.1)
plot([hist1, hist2], ["beta: 0.15", "beta: 0.2"])
plt.show()

print(grad(I)(0.4))