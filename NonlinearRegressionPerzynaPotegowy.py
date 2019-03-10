import csv
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import math
from sklearn.metrics import r2_score
import tensorflow as tf

csv_fname = "10aYieldStress.csv"


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


def LinearRegession(log_funkcja_nadwyzki, log_strain_rate):
    m = GEKKO()
    m.options.IMODE = 2

    x = m.Param(value=log_funkcja_nadwyzki)
    a = m.FV(value=0.00001)
    a.STATUS = 1
    b = m.FV(value=0.00001)
    b.STATUS = 1
    y = m.CV(value=log_strain_rate)
    y.FSTATUS = 1

    m.Equation(y == a * x + b)
    m.solve(disp=False)
    m.solve(disp=False)

    #print('wartosc paramateru n:')
    #print(a.value[0])

    m.options.IMODE = 2
    m.solve(disp=False)

    #print('wartosc paramateru gamma:')
    log_gamma = b.value[0]
    gamma = math.exp(log_gamma)
    #print(gamma)

    from scipy import stats
    slope, intercept, r_value, p_value, \
    std_err = stats.linregress(log_strain_rate, y)
    r2 = r_value ** 2
    cR2 = "R^2 correlation = " + str(r_value ** 2)
    print(cR2)

    return [gamma, a.value[0]]

def NeuralLinearregression(log_funkcja_nadwyzki, log_strain_rate):
    # model parametrs
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.000001)
    train = optimizer.minimize(loss)
    # training data
    x_train = log_funkcja_nadwyzki
    y_train = log_strain_rate

    # training loop

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaulate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    Neuralgamma = math.exp(curr_b)
    Neuraln = curr_W

    return [Neuralgamma, Neuraln, curr_loss]


def ShowResult( gamma, n, strain_rate, yield_stress):
    test_strain_rate = np.arange(0, 100000, 0.1)
    sigma = []
    for i in range(len(test_strain_rate)):
        sigma.append(ksi * (1 + (test_strain_rate[i] / gamma) ** (1 / n)))

    plt.loglog(test_strain_rate, sigma, 'b')
    plt.loglog(strain_rate, yield_stress, 'or')
    plt.xlabel('Strain rate')
    plt.ylabel('Yield stress')
    plt.show()


ReadData(csv_fname)

yield_stress = ReadData(csv_fname)[:,0]
strain_rate = ReadData(csv_fname)[:,1]

#PlotData(strain_rate,yield_stress)

# yield stres przy e_dot = 0.001
ksi=yield_stress[0]

#print(math.log10(100))

log_funkcja_nadwyzki=[]
for i in range(len(yield_stress)):
    log_funkcja_nadwyzki.append(math.log1p((yield_stress[i] / ksi) - 1))

print(" ln funkcja nadwyzki: ")
print(log_funkcja_nadwyzki)

log_strain_rate = []
for i in range(len(strain_rate)):
    log_strain_rate.append(math.log1p(strain_rate[i]))

print("ln strain rate: ")
print(log_strain_rate)

val = LinearRegession(log_funkcja_nadwyzki, log_strain_rate)
#print(val)

gamma= val[0]
n = val[1]

foo =  NeuralLinearregression(log_funkcja_nadwyzki, log_strain_rate)

Neuralgamma=foo[0]
Neuraln=foo[1]
NeuralFitCoeff = foo[2]

ShowResult(gamma, n,strain_rate, yield_stress)

ShowResult(Neuralgamma, Neuraln,strain_rate, yield_stress)

print('Wartości , gamma = ' + str(gamma))
print('Wartości , n = ' + str(n))
