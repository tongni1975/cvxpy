import json
import numpy as np
from time import time
from uuid import uuid4
from twisted.internet.protocol import Protocol, Factory
from twisted.internet.task import LoopingCall
from consensus import prox_step
import cvxpy.settings as s

PING_DELAY = 6   # How often to ping peers
generate_nodeid = lambda: str(uuid4())

class ADMMProtocol(Protocol):
	"""Basic P2P protocol for discovering peers"""
	def __init__(self, factory):
		self.factory = factory
		self.state = "HELLO"
		self.remote_nodeid = None
		self.nodeid = self.factory.nodeid
		self.lc_ping = LoopingCall(self.sendPing)
		self.lastping = None
		
		self.prox = self.factory.prox
		self.v = self.factory.v
		self.rho = self.factory.rho
	
	def connectionMade(self):
		print("Connection from", self.transport.getPeer())
	
	def connectionLost(self, reason):
		if self.remote_nodeid in self.factory.peers:
			self.factory.peers.pop(self.remote_nodeid)
			self.lc_ping.stop()
		print(self.nodeid, "disconnected")
	
	def dataReceived(self, data):
		for line in data.splitlines():
			line = line.strip()
			msgtype = json.loads(line)["msgtype"]
			if self.state == "HELLO" or msgtype == "hello":
				self.handleHello(line)
				self.state = "READY"
			elif msgtype == "ping":
				self.handlePing()
			elif msgtype == "pong":
				self.handlePong()
			elif msgtype == "prox":
				self.handleProx(line)
			elif msgtype == "getprox":
				self.handleGetProx()
	
	def sendHello(self):
		hello = json.dumps({'nodeid': self.nodeid, 'msgtype': 'hello'})
		self.transport.write((hello + "\n").encode("raw_unicode_escape"))
	
	def sendPing(self):
		ping = json.dumps({'nodeid': self.nodeid, 'msgtype': 'ping'})
		print("Pinging", self.remote_nodeid)
		self.transport.write((ping + "\n").encode("raw_unicode_escape"))
	
	def sendPong(self):
		pong = json.dumps({'nodeid': self.nodeid, 'msgtype': 'pong'})
		self.transport.write((pong + "\n").encode("raw_unicode_escape"))
	
	def sendProx(self, xvals):
		for key, value in xvals.items():
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
			self.lc_ping.start(PING_DELAY)
			self.sendGetProx()
	
	def handlePing(self):
		self.sendPong()
	
	def handlePong(self):
		print("Got pong from", self.remote_nodeid)
		self.lastping = time()
	
	def handleProx(self, pres):
		print("Got updated proximal values from", self.remote_nodeid)
		
		# Deserialize by converting lists back to numpy arrays
		pres = json.loads(pres)
		for key, value in pres["xvals"].items():
			if isinstance(value, list):
				pres["xvals"][key] = np.asarray(value)
		
		# Keys (variable ids) were cast to string during serialization
		# and must be converted back to integer
		pres["xvals"] = dict((int(key), value) for key, value in pres["xvals"].items())
		
		# Calculate x_bar^(k+1).
		for key, value in pres["xvals"].items():
			if key in self.v.keys():
				self.v[key]["xbar"].value = (self.v[key]["x"].value + value)/2.0
		
		# Update u^(k+1) += x_bar^(k+1) - x^(k+1).
		for key in self.v.keys():
			self.v[key]["u"].value += (self.v[key]["x"] - self.v[key]["xbar"]).value
		
		# Print current primal values.
		for key, value in self.v.items():
			print("Primal Variable", key, ":\n", value["x"].value)
			print("Dual Variable", key, ":\n", value["u"].value)
	
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
	def __init__(self, prob, rho = 1.0):
		self.prox, self.v, self.rho = prox_step(prob, rho)
	
	def startFactory(self):
		self.peers = {}
		self.nodeid = generate_nodeid()
	
	def buildProtocol(self, addr):
		return ADMMProtocol(self)

def gotProtocol(p):
	"""The callback to start the protocol exchange. 
	   We let connecting nodes begin the hello handshake"""
	p.sendHello()
