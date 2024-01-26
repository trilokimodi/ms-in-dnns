import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# Part A - Make a plot


# Train data
def CreateData(nPi):
    xTrain = np.linspace(0, nPi * np.pi, num=15)
    noiseData = np.random.normal(scale=0.1, size=xTrain.size)
    sinData = np.sin(xTrain)
    yTrain = np.sin(xTrain) + noiseData

    # Test data
    xTest = np.linspace(0, nPi * np.pi, num=10)
    noiseDataTest = np.random.normal(scale=0.1, size=xTest.size)
    sinDataTest = np.sin(xTest)
    yTest = sinDataTest
    return xTrain, yTrain, xTest, yTest, sinData


xTrain, yTrain, xTest, yTest, sinData = CreateData(2)

# plot train and original curve
fig1, ax1 = plt.subplots()
ax1.scatter(xTrain, yTrain, label="Training data", c="r")
ax1.plot(xTrain, sinData, label="Sine Curve", color="b")
# ax = plt.gca()
ax1.xaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: "{:.0f}$\pi$".format(val / np.pi) if val != 0 else "0")
)
ax1.xaxis.set_major_locator(MultipleLocator(base=np.pi))
ax1.legend()
ax1.set_title("Q2 A - Sin curve with 15 training data points")
fig1.savefig("Q2A.png")
# fig1.show()

fig6, ax6 = plt.subplots()


# Part B - Polynomial regression
def fit_poly(x_train, y_train, k):
    # Fits a LSE polynomial fit
    weights = np.polynomial.polynomial.Polynomial.fit(x_train, y_train, deg=k)
    yPredict = np.polyval(p=np.flip(weights.convert().coef), x=x_train)
    weights = np.flip(weights.convert().coef).reshape((1, k + 1))

    # plot train and original curve
    ax6.scatter(x_train, y_train, label="Training data", c="r")
    ax6.plot(x_train, sinData, label="Sine Curve", color="b")
    ax6.plot(x_train, yPredict, label=f"Fitted {k} order polynomial", color="g")
    ax6.legend()
    ax6.set_title("Q2 B - Fitted polynomial curve")
    fig6.savefig("Q2B.png")
    # plt.show()

    return weights


# Part B - Polynomial regression
def fit_poly_no_plot(x_train, y_train, k):
    # Fits a LSE polynomial fit
    weights = np.polynomial.polynomial.Polynomial.fit(x_train, y_train, deg=k)
    weights = np.flip(weights.convert().coef).reshape((1, k + 1))

    return weights


def mse_poly(x, y, w):
    yPredictTest = np.polyval(p=w.flatten(), x=x)
    mse = (np.square(yPredictTest - y)).mean()

    return mse


w = fit_poly(xTrain, yTrain, 3)
mse = mse_poly(xTest, yTest, w)

ax6.legend()
ax6.set_title(f"Q2 B - Fitted polynomial curve with MSE = {mse:.{3}}")
fig6.savefig("Q2BMSE.png")
# plt.show()


# Part C
def Overfit(maxK):
    xTrain, yTrain, xTest, yTest, _ = CreateData(4)
    mse_k = list()
    for iK in range(1, maxK):
        w = fit_poly_no_plot(xTrain, yTrain, iK)
        mse_k.append(mse_poly(xTest, yTest, w))

    return np.log(mse_k), xTrain, yTrain


# Q2 - Overfit contdd
def Q2C():
    maxRuns = 100
    maxK = 15
    leastMSE = np.zeros(maxK - 1)
    for _ in range(maxRuns):
        leastMSEDummy, xTrain, yTrain = Overfit(maxK)
        leastMSE += leastMSEDummy

    leastMSE /= maxRuns

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, 15), leastMSE, label="Log(MSE)")
    ax2.legend()
    ax2.set_title(f"Q2 C - Log MSE plot for different degree and averaged over {maxRuns} runs")
    fig2.savefig("Q2C.png")
    plt.show()

    bestK = leastMSE.argmin()
    bestK += 1
    print(bestK)

    weights = np.polynomial.polynomial.Polynomial.fit(xTrain, yTrain, deg=bestK)
    yPredict = np.polyval(p=np.flip(weights.convert().coef), x=xTrain)

    fig3, ax3 = plt.subplots()
    ax3.plot(xTrain, yPredict, label=f"Fitted {bestK} order polynomial", color="g")
    ax3.legend()
    ax3.set_title("Q2 C - Best K Fit")
    fig3.savefig("Q2CBestKFit.png")
    plt.show()


Q2C()


# Q2 - D Ridge regression
def Q2D():
    maxK = 21
    mse_k_poly = np.zeros((maxK, len(10 ** np.linspace(-5, 0, 20))))
    xTrain, yTrain, xTest, yTest, _ = CreateData(4)
    for k in range(1, maxK):
        for ilamb, lamb in enumerate(10 ** np.linspace(-5, 0, 20)):
            w = ridge_fit_poly(xTrain, yTrain, k, lamb)
            xPoly = PolynomialFeatures(degree=k, include_bias=True)
            if k == 1:
                xTest = xPoly.fit_transform(xTest.reshape(-1, 1))
            else:
                xTest = xPoly.fit_transform(xTest)
            mse_k_poly[k - 1, ilamb] = mse_poly(xTest, yTest, w)

    fig5, ax5 = plt.Figure()
    ax5.imshow(mse_k_poly)
    plt.show()


def ridge_fit_poly(x_train, y_train, k, lamb):
    xPoly = PolynomialFeatures(degree=k, include_bias=True)
    if k == 1:
        x_train = xPoly.fit_transform(x_train.reshape(-1, 1))
    else:
        x_train = xPoly.fit_transform(x_train)

    RidgeRegression = Ridge(alpha=lamb)
    RidgeRegression.fit(x_train, y_train)
    print(RidgeRegression.coef_)

    yPredictTest = RidgeRegression.predict()

    return RidgeRegression.coef_.reshape((1, k + 1))


def mse_sklearn(x, y, w):
    yPredictTest = predict(x_poly)
    mean_squared_error()


# Q2D()
