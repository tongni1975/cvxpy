"""
Copyright 2018 Anqi Fu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe

from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import sum_squares, norm
from consensus_accel import *
from anderson import anderson_accel

def run_consensus(pipes, xbars, udicts, max_iter, accel = False, aa_args = {"m": 5, "max_iter": 10}):
	xids = xbars.keys()
	uids = [udict.keys() for udict in udicts]
	xuarr, xshapes, ushapes = dicts_to_arr(xbars, udicts)
	
	if accel:
		stopping = aa_args.pop("stopping", True)
		g_args = aa_args.pop("g_args", ())   # consensus_map arguments
		g_args += (pipes, xids, uids, xshapes, ushapes)
		def fixed_map(xuarr, i):
			return anderson_accel(consensus_map, xuarr, g_args=g_args + (i,), stopping=stopping, **aa_args)
	else:
		def fixed_map(xuarr, i):
			return consensus_map(xuarr, pipes, xids, uids, xshapes, ushapes, i)
	
	# Fixed point iterations
	resid = np.zeros((max_iter, 2))
	for i in range(max_iter):
		xuarr, rdict = fixed_map(xuarr, i)
		resid[i,:] = rdict["residuals"]
		if i > 0 and rdict["stopped"]:
			break
	
	xbars_f, udicts_f = arr_to_dicts(xuarr, xids, uids, xshapes, ushapes)
	return xbars_f, udicts_f, resid[:(i+1),:]

def plot_residuals(resid, resid_accel = None):
	iters = range(1, resid.shape[0] + 1)
	plt.plot(iters, resid[:,0], label = "Primal")
	plt.plot(iters, resid[:,1], label = "Dual")
	if resid_accel is not None:
		iters_accel = range(1, resid_accel.shape[0] + 1)
		plt.plot(iters_accel, resid_accel[:,0], '--', label = "AA Primal")
		plt.plot(iters_accel, resid_accel[:,1], '--', label = "AA Dual")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Residual")
	plt.show()

def test_basic():
	def g(x):
		x = x[:,0]
		y0 = (2*x[0] + x[0]**2 - x[1])/2.0
		y1 = (2*x[0] - x[0]**2 + 8)/9 + (4*x[1]-x[1]**2)/4.0
		return np.array([[y0, y1]]).T

	m = 5
	x0 = np.array([[-1.0, 1.0]]).T
	res = anderson_accel(g, x0, m, max_iter=10, rcond=2)
	print(res)   # [-1.17397479  1.37821681]

def test_consensus():
	np.random.seed(1)
	m = 100
	n = 10
	MAX_ITER = 10
	x = Variable(n)
	y = Variable(n/2)

	# Problem data.
	alpha = 0.5
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	b = A.dot(xtrue) + np.random.randn(m)

	# List of all the problems with objective f_i.
	p_list = [Problem(Minimize(sum_squares(A*x-b)), [norm(x,2) <= 1]),
			  Problem(Minimize((1-alpha)*sum_squares(y)/2))]
	N = len(p_list)
	rho_init = N*[1.0]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = worker_map, args = (remote, p_list[i], rho_init[i]))]
		procs[-1].start()
	
	# Initial guesses.
	xbars = {x.id: np.zeros(x.size), y.id: np.zeros(y.size)}
	udicts = [{x.id: np.zeros(x.size)}, {y.id: np.zeros(y.size)}]
	
	xbars, udicts, resid = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = False)
	xbars_aa, udicts_aa, resid_aa = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = True)
	[p.terminate() for p in procs]
	
	# Print variable results.
	mse = 0
	for xid in xbars.keys():
		print("Variable %d:\n" % xid, xbars[xid])
		print("Accel Variable %d:\n" %xid, xbars_aa[xid])
		mse += np.sum(np.square(xbars[xid] - xbars_aa[xid]))
	print("Total MSE: %f" % mse)
	
	# Plot residuals.
	plot_residuals(resid[1:,:], resid_aa[1:,:])

def test_ols():
	np.random.seed(1)
	N = 2
	m = N*100
	n = 10
	MAX_ITER = 100
	x = Variable(n)
	
	# Problem data.
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	b = A.dot(xtrue) + np.random.randn(m)
	A_split = np.split(A, N)
	b_split = np.split(b, N)
	
	# List of all the problems with objective f_i.
	p_list = []
	for A_sub, b_sub in zip(A_split, b_split):
		p_list += [Problem(Minimize(sum_squares(A_sub*x-b_sub)))]
	rho_init = N*[0.5]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = worker_map, args = (remote, p_list[i], rho_init[i]))]
		procs[-1].start()
	
	# Initial guesses.
	xbars = {x.id: np.zeros(x.size)}
	udicts = N*[{x.id: np.zeros(x.size)}]
	
	xbars, udicts, resid = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = False)
	# xbars_aa, udicts_aa, resid_aa = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = True)
	[p.terminate() for p in procs]
	
	# Print variable results.
	mse = 0
	for xid in xbars.keys():
		print("Variable %d:\n" % xid, xbars[xid])
		# print("Accel Variable %d:\n" %xid, xbars_aa[xid])
		# mse += np.sum(np.square(xbars[xid] - xbars_aa[xid]))
	# print("Total MSE: %f" % mse)
	
	# Plot residuals.
	plot_residuals(resid[1:,:])
	# plot_residuals(resid[1:,:], resid_aa[1:,:])

def test_lasso():
	np.random.seed(1)
	m = 1000
	n = 10
	MAX_ITER = 30
	DENSITY = 0.75
	x = Variable(n)
	
	# Problem data.
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace = False)
	for idx in idxs:
		xtrue[idx] = 0
	b = A.dot(xtrue) + np.random.randn(m)
	
	# List of all problems with objective f_i.
	p_list = [Problem(Minimize(sum_squares(A*x-b))),
			  Problem(Minimize(norm(x,1)))]
	N = len(p_list)
	rho_init = N*[1.0]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = worker_map, args = (remote, p_list[i], rho_init[i]))]
		procs[-1].start()
	
	# Initial guesses.
	xbars = {x.id: np.zeros(x.size)}
	udicts = N*[{x.id: np.zeros(x.size)}]
	
	xbars, udicts, resid = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = False)
	xbars_aa, udicts_aa, resid_aa = run_consensus(pipes, xbars, udicts, MAX_ITER, accel = True, 
												  aa_args = {"m": 2, "max_iter": 10})
	[p.terminate() for p in procs]
	
	# Print variables results.
	mse = 0
	for xid in xbars.keys():
		print("Variable %d:\n" % xid, xbars[xid])
		print("AA Variable %d:\n" % xid, xbars_aa[xid])
		mse += np.sum(np.square(xbars[xid] - xbars_aa[xid]))
	print("Total MSE: %f" % mse)
	print("Iterations: %d" % resid.shape[0])
	print("AA Iterations: %d" % resid_aa.shape[0])
	
	# Plot residuals.
	# plot_residuals(resid[1:,:])
	plot_residuals(resid[1:,:], resid_aa[1:,:])

print("Basic Test:")
test_basic()

# print("Consensus Test:")
# test_consensus()

# print("OLS Test:")
# test_ols()

print("Lasso Test:")
test_lasso()
