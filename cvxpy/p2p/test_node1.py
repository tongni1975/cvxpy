import sys
import numpy as np
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import sum_squares

from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ServerEndpoint, TCP4ClientEndpoint, connectProtocol
from node_protocol import ADMMProtocol, ADMMFactory, gotProtocol

PORT_LIST = [5998, 5999]

def printError(failure):
	sys.stderr.write(str(failure))

def main():
	np.random.seed(1)
	# m = 100
	# n = 10
	# A = np.random.randn(m*n).reshape(m,n)
	# b = np.random.randn(m)
	
	# Import and split problem data.
	data = np.load("data/test_data.npy")
	A = data[:100,:-1]
	b = data[:100,-1]
	m, n = A.shape
	
	# Pass private problem to ADMM.
	x = Variable((n,), var_id = 1)
	prob = Problem(Minimize(sum_squares(A*x-b)))	
	factory = ADMMFactory(prob, rho = 0.5, verbose = True)
	factory.startFactory()
	
	"""This starts the server-side protocol on port 5999"""
	endpoint = TCP4ServerEndpoint(reactor, 5999)
	endpoint.listen(factory)
	
	"""This starts the client-side protocol"""
	for port in PORT_LIST:
		point = TCP4ClientEndpoint(reactor, "localhost", port)
		d = connectProtocol(point, ADMMProtocol(factory))
		d.addCallback(gotProtocol)
		d.addErrback(printError)
	reactor.run()
		
# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
