import json
from uuid import uuid4
from twisted.internet.protocol import Protocol, Factory
from twisted.internet.task import LoopingCall

PEERS_DELAY = 10   # How often to update peers
generate_nodeid = lambda: str(uuid4())

class NewProtocol(Protocol):
	def __init__(self, factory):
		self.factory = factory
		self.state = "HELLO"
		self.tracker_nodeid = None
		self.nodeid = self.factory.nodeid
		self.lc_peers = LoopingCall(self.sendGetPeers)
	
	def connectionMade(self):
		print("Connection from", self.transport.getPeer())
	
	def connectionLost(self, reason):
		if self.tracker_nodeid in self.factory.peers:
			self.factory.peers.pop(self.tracker_nodeid)
			self.lc_peers.stop()
			print("Tracker", self.tracker_nodeid, "disconnected")
	
	def dataReceived(self, data):
		for line in data.splitlines():
			line = line.strip()
			msgtype = json.loads(line)["msgtype"]
			if self.state == "HELLO":
				self.sendNewPeer()
				self.state = "WAITING"
			elif self.state == "WAITING" and msgtype == "ack":
				print("Tracker", self.tracker_nodeid, "acknowledged us")
				self.tracker_nodeid = line["nodeid"]
				# self.sendGetPeers()
				self.lc_peers.start(PEERS_DELAY)
				self.state = "TRACKED"
			elif msgtype == "peers":
				self.handlePeers(line)
	
	def sendNewPeer(self):
		msg = json.dumps({'nodeid': self.nodeid, 'msgtype': 'newpeer'})
		print("Announcing self to", self.tracker_nodeid)
		self.transport.write((msg + "\n").encode("raw_unicode_escape"))
	
	def sendGetPeers(self):
		msg = json.dumps({'nodeid': self.nodeid, 'msgtype': 'getpeers'})
		print("Requesting peers from tracker", self.tracker_nodeid)
		self.transport.write((msg + "\n").encode("raw_unicode_escape"))
	
	def handlePeers(self, msg):
		print("Got peers from tracker", self.tracker_nodeid)
		msg = json.loads(msg)
		self.factory.peers = msg["peers"]
		
		print("Shared Variable IDs")
		for nodeid, peer in self.factory.peers:
			print("Peer", nodeid, ":", peer["varids"])

class NewFactory(Factory):
	def startFactory(self):
		self.peers = {}
		self.nodeid = generate_nodeid()
	
	def buildProtocol(self, addr):
		return NewProtocol(self)

def gotProtocol(p):
	"""The callback to announce a new node to the tracker."""
	p.sendNewPeer()
