Integrate

Handles Lorenz systems
 > The Lorenz system is a system of ordinary differential equations first studied by Edward Lorenz. It is notable for having chaotic solutions for certain parameter values and initial conditions. 
 > In particular, the Lorenz attractor is a set of chaotic solutions of the Lorenz system which, when plotted, resemble a butterfly or figure eight.
 
 * tf.contrib.integrate.odeint(func,y0,t,rtol=1e-06,atol=1e-12,method=None,options=None,full_output=False,name=None) // Integrates a system of ODE [dy/dt = func(y, t), y(t[0]) = y0]
 
 # solve `dy/dt = -y`, corresponding to exponential decay
tf.contrib.integrate.odeint(lambda y, _: -y, 1.0, [0, 1, 2])
=> [1, exp(-1), exp(-2)]