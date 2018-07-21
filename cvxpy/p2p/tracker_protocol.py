import json
from uuid import uuid4
from twisted.internet.protocol import Protocol, Factory

generate_nodeid = lambda: str(uuid4())

class TrackerProtocol(Protocol):
	"""Protocol for P2P tracker"""
	def __init__(self, factory):
		self.factory = factory
		self.remote_nodeid = None
	
	def connectionMade(self):
		print("Connection from", self.transport.getPeer())
	
	def connectionLost(self, reason):
		if self.remote_nodeid in self.factory.peers:
			self.factory.peers.pop(self.remote_nodeid)
			print(self.remote_nodeid, "disconnected")
	
	def dataReceived(self, data):
		for line in data.splitlines():
			line = line.strip()
			msgtype = json.loads(line)["msgtype"]
			if msgtype == "newpeer":
				self.handleNewPeer(line)
			elif msgtype == "getpeers":
				self.handleGetPeers(line)
	
	def sendTrackerAck(self):
		msg = json.dumps({'nodeid': self.nodeid, 'msgtype': 'ack'})
		self.transport.write((msg + "\n").encode("raw_unicode_escape"))
	
	def sendPeers(self, peers):
		msg = json.dumps({'nodeid': self.nodeid, 'msgtype': 'peers', 'peers': peers})
		self.transport.write((msg + "\n").encode("raw_unicode_escape"))
	
	def handleNewPeer(self, msg):
		msg = json.loads(msg)
		self.remote_nodeid = msg["nodeid"]
		if self.remote_nodeid == self.nodeid:
			print("Connected to self")
			self.transport.loseConnection()
		else:
			remote_peer = self.transport.getPeer()
			self.factory.peers[self.remote_nodeid] = {"ip": remote_peer.ip, "port": remote_peer.port, "varids": set(msg["varids"])}
			self.sendTrackerAck()
	
	def handleGetPeers(self, msg):
		msg = json.loads(msg)
		self.remote_nodeid = msg["nodeid"]
		remote_varids = set(msg["varids"])
		
		# Find all peers with variable(s) in common.
		peers = {}
		for nodeid, node in self.factory.peers:
			vars_shared = node["varids"].intersection(remote_varids)
			if nodeid != self.remote_nodeid and vars_shared:
				peers[nodeid] = {"ip": node["ip"], "port": node["port"], "vars_shared": list(vars_shared)}
		self.sendPeers(peers)

class TrackerFactory(Factory):
	"""Factory for P2P tracker protocol"""
	def startFactory(self):
		self.peers = {}
		self.nodeid = generate_nodeid()
	
	def buildProtocol(self, addr):
		return TrackerProtocol(self)
