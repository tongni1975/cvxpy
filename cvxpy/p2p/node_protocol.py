import json
import numpy as np
from time import time
from uuid import uuid4
from twisted.internet.protocol import Protocol, Factory
from twisted.internet.task import LoopingCall
from consensus import prox_step
import cvxpy.settings as s

PROX_DELAY = 1   # How often to request proximal updates
generate_nodeid = lambda: str(uuid4())

class ADMMProtocol(Protocol):
	"""Basic P2P protocol for discovering peers"""
	def __init__(self, factory):
		self.factory = factory
		self.state = "HELLO"
		self.remote_nodeid = None
		self.nodeid = self.factory.nodeid
		self.verbose = self.factory.verbose
		self.lc_prox = LoopingCall(self.sendGetProx)
		self.lastprox = None
		
		self.prox = self.factory.prox
		self.v = self.factory.v
		self.rho = self.factory.rho
	
	def connectionMade(self):
		print("Connection from", self.transport.getPeer())
	
	def connectionLost(self, reason):
		if self.remote_nodeid in self.factory.peers:
			self.factory.peers.pop(self.remote_nodeid)
			self.lc_prox.stop()
		print(self.nodeid, "disconnected")
	
	def dataReceived(self, data):
		for line in data.splitlines():
			line = line.strip()
			msgtype = json.loads(line)["msgtype"]
			if self.state == "HELLO" or msgtype == "hello":
				self.handleHello(line)
				self.state = "READY"
			elif msgtype == "prox":
				self.handleProx(line)
			elif msgtype == "getprox":
				self.handleGetProx()
	
	def sendHello(self):
		hello = json.dumps({'nodeid': self.nodeid, 'msgtype': 'hello'})
		self.transport.write((hello + "\n").encode("raw_unicode_escape"))
	
	def sendProx(self, xvals):
		for key, value in xvals.items():
			if not isinstance(key, int):
				raise RuntimeError("All variable ids must be integers")
			if isinstance(value, np.ndarray):
				xvals[key] = value.tolist()
		
		prox = json.dumps({'nodeid': self.nodeid, 'msgtype': 'prox', 
						   'status': self.prox.status, 'xvals': xvals})
		print("Sending updated proximal values to peers")
		self.transport.write((prox + "\n").encode("raw_unicode_escape"))
	
	def sendGetProx(self):
		getprox = json.dumps({'nodeid': self.nodeid, 'msgtype': 'getprox'})
		print("Requesting proximal update from peers")
		self.transport.write((getprox + "\n").encode("raw_unicode_escape"))
		
	def handleHello(self, hello):
		hello = json.loads(hello)
		self.remote_nodeid = hello["nodeid"]
		if self.remote_nodeid == self.nodeid:
			print("Connected to self")
			self.transport.loseConnection()
		else:
			self.factory.peers[self.remote_nodeid] = self
			self.lc_prox.start(PROX_DELAY)
	
	def handleProx(self, pres):
		print("Got updated proximal values from", self.remote_nodeid)
		self.lastprox = time()
		
		# Deserialize by converting lists back to numpy arrays
		pres = json.loads(pres)
		for key, value in pres["xvals"].items():
			if isinstance(value, list):
				pres["xvals"][key] = np.asarray(value)
		
		# Keys (variable ids) were cast to string during serialization
		# and must be converted back to integer
		pres["xvals"] = dict((int(key), value) for key, value in pres["xvals"].items())
		
		# Calculate x_bar^(k+1).
		xbar_old = dict((key, vdict["xbar"].value) for key, vdict in self.v.items())
		for key, value in pres["xvals"].items():
			if key in self.v.keys():
				self.v[key]["xbar"].value = (self.v[key]["x"].value + value)/2.0
		
		# Update u^(k+1) += x_bar^(k+1) - x^(k+1).
		res_ssq = {"primal": 0, "dual": 0}
		for key in self.v.keys():
			self.v[key]["u"].value += (self.v[key]["x"] - self.v[key]["xbar"]).value
			
			if self.v[key]["x"].value is None:
				primal = -self.v[key]["xbar"].value
			else:
				primal = (self.v[key]["x"] - self.v[key]["xbar"]).value
			dual = (self.rho*(self.v[key]["xbar"] - xbar_old[key])).value
			res_ssq["primal"] += np.sum(np.square(primal))
			res_ssq["dual"] += np.sum(np.square(dual))
		
		# Print sum-of-squared primal/dual residuals.
		if self.verbose:
			print("Primal Residual:", res_ssq["primal"])
			print("Dual Residual:", res_ssq["dual"])
			# for key, vdict in self.v.items():
			#	print("Variable", key)
			#	print("Primal:", vdict["x"].value)
			#	print("Dual:", vdict["u"].value)
	
	def handleGetProx(self):
		print("Got proximal update request from", self.remote_nodeid)
		
		# Proximal step for x^(k+1).
		self.prox.solve()
		
		# Check if proximal step converged.
		if self.prox.status in s.INF_OR_UNB:
			raise RuntimeError("Proximal problem is infeasible or unbounded")
		
		# Send x^{k+1} to peer.
		xvals = {}
		for xvar in self.prox.variables():
			xvals[xvar.id] = xvar.value
		self.sendProx(xvals)

class ADMMFactory(Factory):
	"""Factory for basic P2P protocol"""
	def __init__(self, prob, rho = 1.0, verbose = False):
		self.verbose = verbose
		self.prox, self.v, self.rho = prox_step(prob, rho)
	
	def startFactory(self):
		self.peers = {}
		self.nodeid = generate_nodeid()
		self.prox.solve()   # Solve proximal problem to get initial x^0.
	
	def buildProtocol(self, addr):
		return ADMMProtocol(self)

def gotProtocol(p):
	"""The callback to start the protocol exchange. 
	   We let connecting nodes begin the hello handshake"""
	p.sendHello()
