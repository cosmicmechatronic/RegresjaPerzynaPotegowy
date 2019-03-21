import csv
import numpy as np
import matplotlib.pyplot as plt
dt = np.dtype(np.float64)
from mpmath import mp
from gekko import GEKKO
import math
from mpmath import mp
import tensorflow as tf

csv_fname = "05aYieldStress.csv"

# initializing the titles and rows list
def ReadData (csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        # get header from first row
        # headers = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data).astype(float)

        #print('wiersze:, kolumny:')
        #print(data.shape)
        #print('test danych')
        #print(data[:3])

    return(data)

def PlotData(a_data, b_data):
    #plt.scatter(a_data, b_data)
    plt.loglog(a_data, b_data)
    plt.xlabel('Strain rate')
    plt.ylabel('Yield stress')
    plt.show()


def LinearRegession(funkcja_nadwyzki, strain_rate):
    m = GEKKO()
    m.options.IMODE = 2

    x = m.Param(value=funkcja_nadwyzki)
    a = m.FV(value=0.00001)
    a.STATUS = 1
    #b = m.FV(value=0.00001)
    #b.STATUS = 1
    y = m.CV(value=strain_rate)
    y.FSTATUS = 1

    m.Equation(y == a * x)
    m.solve(disp=False)
    m.solve(disp=False)

    #print('wartosc paramateru n:')
    #print(a.value[0])

    m.options.IMODE = 2
    m.solve(disp=False)

    #print('wartosc paramateru gamma:')
    gamma = a.value[0]

    from scipy import stats
    slope, intercept, r_value, p_value, \
    std_err = stats.linregress(strain_rate, y)

    r2 = r_value ** 2
    cR2 = "R^2 correlation = " + str(r_value ** 2)
    print(cR2)


    return [gamma]


def NeuralLinearRegresion(funkcja_nadwyzki, strain_rate):
    # model parameters
    W = tf.Variable([.3], tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.000001)
    train = optimizer.minimize(loss)
    # training data
    x_train = funkcja_nadwyzki
    y_train = strain_rate
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W,  curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
    #print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


    gamma = curr_W
    return[gamma, curr_loss]

def ShowResult( gamma,  strain_rate, yield_stress, ksi):
    test_strain_rate = np.arange(0, 100000, 0.1)
    sigma = []
    for i in range(len(test_strain_rate)):
        sigma.append(ksi * (1 + (mp.log(1+ (test_strain_rate[i] / gamma) ) ) ))

    plt.loglog(test_strain_rate, sigma, 'b')
    plt.loglog(strain_rate, yield_stress, 'or')
    plt.xlabel('Strain rate')
    plt.ylabel('Yield stress')
    plt.show()




ReadData(csv_fname)

yield_stress = ReadData(csv_fname)[:,0]
strain_rate = ReadData(csv_fname)[:,1]


# yield stres przy e_dot = 0.001
ksi=yield_stress[0]

wykladnik = []
for i in range (len(yield_stress)):
    wykladnik.append((yield_stress[i]/ksi)-1)

funkcja_nadwyzki=[]
for i in range(len(wykladnik)):
    funkcja_nadwyzki.append(mp.exp(wykladnik[i])- 1)

val = LinearRegession(funkcja_nadwyzki, strain_rate)
gamma= val[0]
foo = NeuralLinearRegresion(funkcja_nadwyzki, strain_rate)

print( "wspolczynnik gamma")
print(gamma)
ShowResult(gamma, strain_rate, yield_stress, ksi)
neural_gamma = foo[0]
print( "wspolczynnik neural gamma")
print(neural_gamma)
ShowResult(neural_gamma, strain_rate, yield_stress, ksi)
