import numpy as np 
from BaysianOptimizer import BaysianMaximization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytest

def model(a,b,c,d, x = 0, noise = 0.05):
    try:
        noise_term = +noise*np.random.randn(len(x))
    except:
        noise_term = +noise*np.random.randn()

    return (x**2 * np.sin(5 * np.pi * x)**6.0) + noise_term

def FrankeFunction(a,b,c,d,x= 0,y = 0, noise = 0.1):
    try:
        noise_term = +noise*np.random.randn(len(x))
    except:
        noise_term = +noise*np.random.randn()

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise_term

def test_optimizer():
    opt = BaysianMaximization(model, {},{'x':(0,1)})
    fill = np.random.rand(100).reshape((25,4))

    opt.SetData(fill, fill)
    opt.InitialGuesses(50)
    opt.OptimizeHyperPar(cycles=50)

    best_opt = opt.best_model_kargs['x']

    x = np.linspace(0,1,500)
    y = model(0,0,0,0, x=x , noise = 0)
    y_gp, sd = opt.gp_estimator.predict(x.reshape(len(x),1), return_std=True)
    actual_opt = x[np.argmax(y)]

    plt.figure(figsize =(10,10))
    plt.subplot(1,2,1)
    plt.scatter(opt.hyperpar,opt.model_score, label = 'sample points')
    plt.plot(x, y, label='true function')
    plt.fill_between(x, y_gp -sd, y_gp + sd, alpha =0.4, label = 'GP', color ='tab:orange')
    plt.plot(x, y_gp, color ='tab:orange')
    plt.ylabel("f(x)", fontsize=28)
    plt.xlabel("x", fontsize = 28)
    plt.legend(loc='best', fontsize =24)
    plt.tick_params(size = 20, labelsize=22)


    plt.subplot(1,2,2)
    plt.hist(opt.hyperpar, bins = 20)
    plt.xlabel("x", fontsize = 28)
    plt.ylabel("count", fontsize=28)
    plt.legend(loc='best', fontsize =24)
    plt.tick_params(size = 20, labelsize=22)
    plt.tight_layout()
    plt.savefig('Results/BaysianOpt/test_1D.pdf')

    print("found best", best_opt)
    print("Actual best", actual_opt)
    #assert  pytest.approx(abs(best_opt - actual_opt) == 0,abs =1e-3)

def test_optimizer2D():
    opt = BaysianMaximization(FrankeFunction, {},{'x':(0,1), 'y':(0,1)})
    fill = np.random.rand(100).reshape((25,4))

    opt.SetData(fill, fill)
    opt.InitialGuesses(200)
    opt.OptimizeHyperPar(cycles=200)

    best_opt_x = opt.best_model_kargs['x']
    best_opt_y = opt.best_model_kargs['y']

    x = np.linspace(0,1,500)
    y = np.linspace(0,1,500)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(0,0,0,0, x=X , y= Y, noise = 0)
    y_gp = opt.gp_estimator.predict(np.transpose([X.flatten(),Y.flatten()]))

    ind = np.unravel_index(np.argmax(Z,axis= None), Z.shape)
    
    plt.subplot(1,2,1)
    ax = plt.axes(projection='3d')
    ax.set_title("Franke + Sample")
    ax.plot_surface(X,Y,Z)
    par = opt.hyperpar.T
    ax.scatter(par[0], par[1], opt.model_score )
    plt.show()

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title("Franke Function", fontsize =28)
    plt.contourf(X,Y,Z, vmin =0, vmax= 1.1*Z.max(), cmap='coolwarm')
    plt.xlabel("x", fontsize = 28)
    plt.xlabel("y", fontsize = 28)
    plt.subplot(2,2,2)
    plt.title("GP", fontsize =28)
    c =plt.contourf(X,Y, y_gp.reshape((500,500)), vmin =0, vmax= 1.1*Z.max(), cmap='coolwarm')
    
    plt.xlabel("x", fontsize = 28)
    plt.xlabel("y", fontsize = 28)
    plt.subplot(2,2,3)
    plt.title("Histogram", fontsize =28)
    d =plt.hist2d(*par)
    dbar=plt.colorbar(d[-1], label='count')
    dbar.ax.tick_params(labelsize =22)
    dbar.ax.set_title('count',fontsize =28)
    plt.plot( X[ind], Y[ind], marker='x', color='r')

    plt.subplot(2,2,4)
    plt.title("Functions", fontsize = 28)
    cbar =plt.colorbar(c, label='f(x,y)')
    cbar.ax.tick_params(labelsize =22)
    cbar.ax.set_title('f(x,y)',fontsize =28)
    plt.tight_layout()
    plt.savefig('Results/BaysianOpt/test_2D.pdf')

    print("found best", (best_opt_x, best_opt_y))
    print("Actual best", X[ind], Y[ind])

test_optimizer2D()
