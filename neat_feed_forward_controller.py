from controller import Controller
import numpy as np
import math

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


class player_controller(Controller):
	def __init__(self, net,write=False):
		self.net = net
		self.move_counter = [0,0,0,0,0]
		self.write = write
		self.f = open("feedforward_keystrokes.txt", "w")
	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
		# print(inputs)
		# print(inputs.shape)
		output = self.net.activate(inputs)
		# print(output)
		# exit()
		if output[0] > 0.5:
			left = 1
			self.move_counter[0]+=1
		else:
			left = 0

		if output[1] > 0.5:
			self.move_counter[1]+=1
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			self.move_counter[2]+=1
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			self.move_counter[3]+=1
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			self.move_counter[4]+=1
			release = 1
		else:
			release = 0
		if self.write:
			self.f.write(str(self.move_counter))		
		return [left, right, jump, shoot, release]


class gen_player_controller(Controller):
	def __init__(self, snets,gnet):
		self.snets = snets
		self.gnet =gnet
	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
		# print(inputs)
		# print(inputs.shape)
		soutputs = []
		for snet in self.snets:

			soutputs.extend(snet.activate(inputs))
		# print(output)
		# print(soutputs)
		goutput = self.gnet.activate(soutputs)

		# exit()
		if goutput[0] > 0.5:
			left = 1
		else:
			left = 0

		if goutput[1] > 0.5:
			right = 1
		else:
			right = 0

		if goutput[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if goutput[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if goutput[4] > 0.5:
			release = 1
		else:
			release = 0
		return [left, right, jump, shoot, release]


class gen_skip_player_controller(Controller):
	def __init__(self, snets,gnet):
		self.snets = snets
		self.gnet =gnet
	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
		# print(inputs)
		# print(inputs.shape)
		soutputs = []
		for snet in self.snets:

			soutputs.extend(snet.activate(inputs))
		# print(output)
		# print(soutputs)
		# print(inputs)
		# print(soutputs.extend(inputs))
		# print(len(chain(soutputs,inputs)))
		# exit()
		goutput = self.gnet.activate(list(soutputs)+list(inputs))

		# exit()
		if goutput[0] > 0.5:
			left = 1
		else:
			left = 0

		if goutput[1] > 0.5:
			right = 1
		else:
			right = 0

		if goutput[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if goutput[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if goutput[4] > 0.5:
			release = 1
		else:
			release = 0
		return [left, right, jump, shoot, release]