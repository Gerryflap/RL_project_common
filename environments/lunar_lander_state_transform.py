import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    return (x - np.mean(x))/(np.var(x)**0.5)


def transform_state(x):
    if x.shape != (8,):
        raise ValueError("Expected one-dimensional numpy array of shape (8,), got: %s"%(str(x.shape)))
    x = x[:-2]
    out = np.zeros((6,))

    # Pos angle
    out[0] = np.round((np.arctan2(x[0], x[1])+ 0.5 * np.math.pi)/np.math.pi, 1)

    # Pos distance
    out[1] = np.round((2*np.sqrt(x[0]**2 + x[1]**2))**(1/3),1)

    # Velocities
    out[2] = np.round(2*np.sqrt(np.abs(x[2])) * np.sign(x[2]),0)
    out[3] = np.round(2*np.sqrt(np.abs(x[3])) * np.sign(x[3]),0)

    # Angle
    out[4] = np.round(2.5*np.sqrt(np.abs(x[4])) * np.sign(x[4]),0)

    # Anglular velocity
    out[5] = np.round(np.sqrt(np.abs(x[5])) * np.sign(x[5]),0)
    return tuple(out)

if __name__ == "__main__":


    # Pos
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-0.3, 1.5, 1000)
    xx, yy = np.meshgrid(x, y)

    angles = np.arctan2(xx, yy)
    distances = (2*np.sqrt(xx**2 + yy**2))**(1/3)

    angles = np.round((angles + 0.5 * np.math.pi)/np.math.pi, 1)
    distances = np.round(distances, 1)

    img = np.zeros(angles.shape + (3,))
    img[:,:,0] = angles
    img[:,:,1] = distances

    plt.title("Pos Angle")
    plt.imshow(angles)
    plt.show()

    plt.title("Pos distance")
    plt.imshow(distances)
    plt.show()

    # Velocity
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y)

    xx = np.sqrt(np.abs(xx)) * np.sign(xx)
    yy = np.sqrt(np.abs(yy)) * np.sign(yy)

    xx = np.round(xx*2, 0)
    yy = np.round(yy*2, 0)

    plt.title("X Velocity")
    plt.imshow(xx)
    plt.show()
    plt.title("Y Velocity")
    plt.imshow(yy)
    plt.show()

    # Angles
    x = np.linspace(-np.math.pi, np.math.pi, 1000)
    y = np.linspace(-2*np.math.pi, 2*np.math.pi, 1000)
    xx, yy = np.meshgrid(x, y)

    xx = np.sqrt(np.abs(xx)) * np.sign(xx)
    yy = np.sqrt(np.abs(yy)) * np.sign(yy)

    xx = np.round(xx*2.5, 0)
    yy = np.round(yy, 0)

    plt.title("Lander Angle")
    plt.imshow(xx)
    plt.show()
    plt.title("Lander Angular Velocity")
    plt.imshow(yy)
    plt.show()