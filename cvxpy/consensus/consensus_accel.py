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

import cvxpy.settings as s
from cvxpy.problems.problem import Problem, Minimize
from cvxpy.expressions.constants import Parameter
from cvxpy.atoms import sum_squares

from consensus import prox_step, x_average, res_stop
from anderson import anderson_accel

import numpy as np
import warnings
from collections import defaultdict

def dicts_to_arr(xbars, udicts):
	# Keep shape information.
	xshapes = [xbar.shape for xbar in xbars.values()]
	ushapes = [[u.shape for u in udict.values()] for udict in udicts]
	
	# Flatten x_bar and u into vectors.
	xflat = [xbar.flatten(order='C') for xbar in xbars.values()]
	uflat = [u.flatten(order='C') for udict in udicts for u in udict.values()]
	
	xuarr = np.concatenate(xflat + uflat)
	xuarr = np.array([xuarr]).T
	return xuarr, xshapes, ushapes

def arr_to_dicts(arr, xids, uids, xshapes, ushapes):
	# Split array into x_bar and u vectors.
	N = len(ushapes)
	xnum = len(xshapes)
	xelems = [np.prod(shape) for shape in xshapes]
	uelems = [[np.prod(shape) for shape in ushape] for ushape in ushapes]
	split_idx = np.cumsum([np.sum(xelems)] + [np.sum(uelem) for uelem in uelems])
	asubs = np.split(arr, split_idx)
	
	# Reshape x_bar.
	sidx = 0
	xbars = []
	for i in range(xnum):
		eidx = sidx + xelems[i]
		xvec = asubs[0][sidx:eidx]
		xbars += [np.reshape(xvec, xshapes[i])]
		sidx += xelems[i]
	
	# Reshape u_i for each pipe.
	udicts = []
	for j in range(N):
		sidx = 0
		uvals = []
		for i in range(len(ushapes[j])):
			eidx = sidx + uelems[j][i]
			uvec = asubs[j+1][sidx:eidx]
			uvals += [np.reshape(uvec, ushapes[j][i])]
			sidx += uelems[j][i]
		udicts += [uvals]
	
	# Compile into dicts.
	xbars = dict(zip(xids, xbars))
	udicts = [dict(zip(uid, u)) for uid, u in zip(uids, udicts)]
	return xbars, udicts

def worker_map(pipe, p, rho_init, *args, **kwargs):
	# Spectral step size parameters.
	spectral = kwargs.pop("spectral", False)
	Tf = kwargs.pop("Tf", 2)
	eps = kwargs.pop("eps", 0.2)
	C = kwargs.pop("C", 1e10)
	
	# Initiate proximal problem.
	prox, v, rho = prox_step(p, rho_init)
	
	# ADMM loop.
	while True:
		# Receive x_bar^(k) and u^(k).
		xbars, uvals, cur_iter = pipe.recv()
		
		ssq = {"primal": 0, "dual": 0, "x": 0, "xbar": 0, "u": 0}
		for key in v.keys():
			# Calculate primal/dual residual.
			if v[key]["x"].value is None:
				primal = xbars[key]
			else:
				primal = (xbars[key] - v[key]["x"]).value
			dual = (rho*(v[key]["xbar"] - xbars[key])).value
			
			# Set parameter values of x_bar^(k) and u^(k).
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value = uvals[key]
			
			# Save stopping rule criteria.
			ssq["primal"] += np.sum(np.square(primal))
			ssq["dual"] += np.sum(np.square(dual))
			if v[key]["x"].value is not None:
				ssq["x"] += np.sum(np.square(v[key]["x"].value))
			ssq["xbar"] += np.sum(np.square(v[key]["xbar"].value))
			ssq["u"] += np.sum(np.square(v[key]["u"].value))
		pipe.send(ssq)
		
		# Proximal step for x^(k+1) with x_bar^(k) and u^(k).
		# v_copy = {key: {key_s: val_s.value for key_s, val_s in val.items()} for key, val in v.items()}
		prox.solve(*args, **kwargs)
		if prox.status == s.OPTIMAL_INACCURATE:
			warnings.warn("Proximal step may be inaccurate.")
		
		# Calcuate x_bar^(k+1).
		xvals = {k: np.asarray(d["x"].value) for k,d in v.items()}
		pipe.send((prox.status, xvals))
		xbars = pipe.recv()
		
		# Update u^(k+1) += rho*(x_bar^(k+1) - x^(k+1)).
		for key in v.keys():
			uvals[key] += rho.value*(xbars[key] - v[key]["x"].value)
			
		# Return u^(k+1) and step size.
		pipe.send(uvals)

def consensus_map(xuarr, pipes, xids, uids, xshapes, ushapes, cur_iter):
	xbars, udicts = arr_to_dicts(xuarr, xids, uids, xshapes, ushapes)
	
	# Scatter x_bar^(k) and u^(k).
	N = len(pipes)
	for i in range(N):
		pipes[i].send((xbars, udicts[i], cur_iter))
	
	# Calculate normalized residuals.
	ssq = [pipe.recv() for pipe in pipes]
	primal, dual, stopped = res_stop(ssq)
	
	# Gather and average x^(k+1).
	prox_res = [pipe.recv() for pipe in pipes]
	xbars_n = x_average(prox_res)
	
	# Scatter x_bar^(k+1).
	for pipe in pipes:
		pipe.send(xbars_n)
	
	# Gather updated u^(k+1).
	udicts_n = [pipe.recv() for pipe in pipes]
	xuarr_n, xshapes, ushapes = dicts_to_arr(xbars_n, udicts_n)
	rdict = {"residuals": np.array([primal, dual]), "stopped": stopped}
	return xuarr_n, rdict
