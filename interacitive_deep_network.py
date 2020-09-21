# import statements
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# build and trian network
np.random.seed(0)
n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=y, verbose=0, batch_size=20, epochs=100, shuffle='true')

# define plots and boundries
def plot_boundry(X, y, model):
    x_span = np.linspace(min(X[:,0])-0.25, max(X[:,0])+0.25, 50)
    y_span = np.linspace(min(X[:, 1])-0.25, max(X[:, 1])+0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    predict_func = model.predict(grid)
    z = predict_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

# define main function
def main():
    x = float(input('Note: Value range should be between 0.0 and 1.5 \nEnter the x coordinate as float:'))
    y = float(input('Note: Value range should be between 0.0 and 1.5 \nEnter the y coordinate as float:'))

    plot_boundry(X, y, model)
    plt.scatter(X[:n_pts,0], X[:n_pts,1])
    plt.scatter(X[n_pts:,0], X[n_pts:,1])
    point = np.array([[x,y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker='o', markersize=10, color='red')
    
    print('Prediction is: ', prediction)
    plt.title('Predicition')
    plt.show()

# call main()    
if __name__ == '__main__':
    main()