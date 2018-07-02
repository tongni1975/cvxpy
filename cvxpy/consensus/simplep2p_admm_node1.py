import sys
import numpy as np
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import sum_squares

from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ServerEndpoint, TCP4ClientEndpoint, connectProtocol
from p2p_protocol_admm import ADMMProtocol, ADMMFactory, gotProtocol

PORT_LIST = [5998, 5999]

def printError(failure):
	sys.stderr.write(str(failure))

def main():
	np.random.seed(1)
	m = 100
	n = 10
	
	x = Variable((n,), var_id = 1)
	A = np.random.randn(m*n).reshape(m,n)
	b = np.random.randn(m)
	prob = Problem(Minimize(sum_squares(A*x-b)))	

	factory = ADMMFactory(prob)
	factory.startFactory()
	
	"""This starts the server-side protocol on port 5999"""
	endpoint = TCP4ServerEndpoint(reactor, 5999)
	endpoint.listen(factory)
	
	"""This starts the client-side protocol"""
	for port in PORT_LIST:
		point = TCP4ClientEndpoint(reactor, "localhost", port)
		d = connectProtocol(point, ADMMProtocol(factory))
		d.addCallback(gotProtocol)
		# d.addErrback(printError)
	reactor.run()
		
# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
