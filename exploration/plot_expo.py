import matplotlib.pyplot as plt
import numpy as np

EPS_START = 0.9  # Starting epsilon value
EPS_END = 0.05    # Final epsilon value
EPS_DECAY = 500  # Number of steps over which to decay epsilon

# Number of steps
num_steps = 2000

def run():
    # Calculate epsilon values
    steps = np.arange(num_steps)
    epsilons = [EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY) for step in steps]

    # Plot epsilon values
    plt.plot(steps, epsilons)
    plt.xlabel('Steps')
    plt.ylabel('Epsilon')
    plt.title('Decay of Epsilon')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()