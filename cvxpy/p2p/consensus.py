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
from cvxpy.atoms import multiply, sum_squares

import numpy as np
from collections import defaultdict

def flip_obj(prob):
	"""Helper function to flip sign of objective function.
	"""
	if isinstance(prob.objective, Minimize):
		return prob.objective
	else:
		return -prob.objective

def prox_step(prob, rho_init, scaled = False):
	"""Formulates the proximal operator for a given objective, constraints, and step size.
	Parikh, Boyd. "Proximal Algorithms."
	
	Parameters
    ----------
    prob : Problem
        The objective and constraints associated with the proximal operator.
        The sign of the objective function is flipped if `prob` is a maximization problem.
    rho_init : float
        The initial step size.
    scaled : logical, optional
    	Should the dual variable be scaled?
    
    Returns
    ----------
    prox : Problem
        The proximal step problem.
    vmap : dict
        A map of each proximal variable id to a dictionary containing that variable `x`,
        the mean variable parameter `xbar`, and the associated dual parameter `y`.
    rho : Parameter
        The step size parameter.
	"""
	vmap = {}   # Store consensus variables
	f = flip_obj(prob).args[0]
	rho = Parameter(value = rho_init, nonneg = True)   # Step size
	
	# Add penalty for each variable.
	for xvar in prob.variables():
		xid = xvar.id
		shape = xvar.shape
		vmap[xid] = {"x": xvar, "xbar": Parameter(shape, value = np.zeros(shape)),
					 "y": Parameter(shape, value = np.zeros(shape))}
		dual = vmap[xid]["y"] if scaled else vmap[xid]["y"]/rho
		f += (rho/2.0)*sum_squares(xvar - vmap[xid]["xbar"] + dual)
	
	prox = Problem(Minimize(f), prob.constraints)
	return prox, vmap, rho

def x_average(prox_res):
	"""Average the primal variables over the nodes in which they are present,
	   weighted by each node's step size.
	"""
	xmerge = defaultdict(list)
	rho_sum = defaultdict(float)
	
	for status, rho, xvals in prox_res:
		# Check if proximal step converged.
		if status in s.INF_OR_UNB:
			raise RuntimeError("Proximal problem is infeasible or unbounded")
		
		# Merge dictionary of x values
		for key, value in xvals.items():
			xmerge[key].append(rho*value)
			rho_sum[key] += rho
	
	return {key: np.sum(np.array(xlist), axis = 0)/rho_sum[key] for key, xlist in xmerge.items()}
