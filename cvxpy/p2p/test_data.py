import numpy as np
from cvxpy import Variable, Problem, Minimize, sum_squares
from cvxconsensus import Problems

def main():
	np.random.seed(1)
	data = np.load("data/test_data.npy")
	A = data[:,:-1]
	b = data[:,-1]
	m, n = A.shape

	x = Variable((n,), var_id = 1)
	prob = Problem(Minimize(sum_squares(A*x-b)))
	prob.solve()
	print("Objective:", prob.value)
	print("Variable:", x.value)
	
	A0 = data[:100,:-1]
	b0 = data[:100,-1]
	prob0 = Problem(Minimize(sum_squares(A0*x-b0)))
	
	A1 = data[100:,:-1]
	b1 = data[100:,-1]
	prob1 = Problem(Minimize(sum_squares(A1*x-b1)))
	
	probs = Problems([prob0, prob1])
	probs.solve(method = "consensus", rho_init = 2*[0.5])
	print("Consensus Objective:", probs.value)
	print("Consensus Variable:", x.value)

if __name__ == '__main__':
    main()
