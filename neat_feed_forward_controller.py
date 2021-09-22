from controller import Controller
import numpy as np
import math

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


class player_controller(Controller):
	def __init__(self, net):
		self.net = net

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
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0
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
		goutput = self.gnet.activate(soutputs.extend(inputs))

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