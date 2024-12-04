import matplotlib.pyplot as plt

def plot_training_curve(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
1