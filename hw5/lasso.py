import numpy as np
import matplotlib.pyplot as plt
import time


def LassoObjective(wplus, wminus, Phi, Y, lmbd):
    ''' evaluates the objective function at (wplus, wminus)
    L2 loss + L1 regularization
    '''
    w = wplus - wminus
    return ((Phi @ w - Y) ** 2).mean(
        ) + lmbd * np.abs(w).sum()


def GradLassoObjective(wplus, wminus, Phi, Y, lmbd):
    ''' computes the gradients of the objective function
    at (wplus, wminus)
    gradwplus: gradient wrt wplus
    gradwminus: gradient wrt minus

    FILL IN
    '''

    gradwplus = np.zeros(wplus.shape)    # TODO 
    gradwminus = np.zeros(wminus.shape)  # TODO 
    return gradwplus, gradwminus


def ProjectionPositiveOrthant(x):
    ''' returns the projection of x onto the positive orthant

    FILL IN
    '''
    y = x # TODO 
    return y


def getStepSize(wplus, wminus, Phi, Y, lmbd, gradwplus,
                gradwminus, loss):
    ''' performs one step of projected gradient descent (i.e.
    compute next iterate) with step size selection via
    backtracking line search

    input
    loss: objective function at current iterate (wplus, wminus)

    output
    wplusnew, wminusnew: next iterates wplus_{t+1}, wminus_{t+1}
    lossnew: objective function at the new iterate
    
    FILL IN
    '''
    alpha, beta, sigma = 1., .1, .1
    wplusnew, wminusnew = wplus.copy(), wminus.copy()
    lossnew = np.float('Inf') # make sure to enter the loop

    # choose the step size alpha with backtracking line search
    while lossnew > loss + sigma * ((gradwplus * (
        wplusnew - wplus)).sum() + (gradwminus * (
        wminusnew - wminus)).sum()):
        # get new step size to test
        alpha *= beta

        # projected gradient step for wplus and wminus with step size alpha
        # i.e. compute x_{t+1} as in the text
        # FILL IN
        print('fill in with projected gradient step')
        wplusnew = wplus    # TODO 
        wminusnew = wminus  # TODO

        # compute new value of the objective
        lossnew = LassoObjective(wplusnew, wminusnew, Phi, Y, lmbd)

    return wplusnew, wminusnew, lossnew


def Lasso(Phi, Y, lmbd):
    ''' compute weight of linear regression with Lasso

    Phi: deisgn matrix n x d
    Y: true values n x 1
    lmbd: weight of regularization

    output: weights of linear regression d x 1
    '''
    # initialize wplus, wminus
    wplus = np.random.rand(Phi.shape[1], 1)
    wminus = np.random.rand(*wplus.shape)
    loss = LassoObjective(wplus, wminus, Phi, Y, lmbd)

    counter = 1
    while counter > 0:
        # compute gradients wrt wplus and wminus
        gradwplus, gradwminus = GradLassoObjective(
            wplus, wminus, Phi, Y, lmbd)

        # compute new iterates
        wplus, wminus, loss = getStepSize(wplus,
            wminus, Phi, Y, lmbd, gradwplus, gradwminus, loss)

        if (counter % 100) == 0:
            # check if stopping criterion is met
            wnew = wplus - wminus
            ind = wnew != 0.
            indz = wnew == 0.
            r = 2 / Phi.shape[0] * (Phi.T @ (Phi @ wnew - Y))
            stop = np.abs(r[ind] + lmbd * np.sign(wnew[ind]
                )).sum() + (np.abs(r[indz]) - lmbd * np.ones_like(
                r[indz])).clip(0.).sum()
            print('iter={} current objective={:.3f} nonzero weights={}'.format(
                counter, loss, ind.sum()) +\
                ' stop={:.5f}'.format(stop / Phi.shape[0]))
            if np.abs(stop) / Phi.shape[0] < 1e-5:
                break
        counter += 1

    #print((wplus == 0).sum(), (wminus == 0).sum())

    return wplus - wminus


def L2Loss(x, y):
    return (x - y) ** 2


if(__name__ == '__main__'):
    lmbd = 10

    data_new = np.load('multidim_data.npy', allow_pickle=True).item()
    X, Y = data_new['Xtrain'], data_new['Ytrain']
    Xval, Yval = data_new['Xtest'], data_new['Ytest']

    # TODO  normalize features

    # TODO add feature for offset

    t_init = time.time()
    w = Lasso(X, Y, lmbd)
    print('runtime: {:.3f} s'.format(time.time() - t_init))

    LossTrain = L2Loss(X @ w, Y)
    print('training loss: {:.5f}'.format(LossTrain.mean()))
    Yvalpred = 0 # TODO, compute predictions
    LossVal = L2Loss(Yvalpred, Yval)
    print('validation loss: {:.5f}'.format(LossVal.mean()))
    plt.plot(Yval, Yvalpred, '.g')
    plt.plot([0, np.amax(Yval)], [0, np.amax(Yval)], '-r')
    ax = plt.gca()
    ax.set_xlabel('true values')
    ax.set_ylabel('predicted values')
    plt.show()