import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plotBubble(xindices, yindices, labels, xticklabels = None, yticklabels = None, offset = 0.25, sort = False):
	coff = 20/9
	assert len(xindices) == len(yindices)
	if sort:
		convx, convy = get_sorted_idx(xindices, yindices)
		xindices = [convx[x] for x in xindices]
		yindices = [convy[y] for y in yindices]

		if xticklabels is not None:
			xticklabels_ = xticklabels[:]
			for i,tl in enumerate(xticklabels): xticklabels_[convx[i]] = tl
			xticklabels = xticklabels_
		if yticklabels is not None:
			yticklabels_ = yticklabels[:]
			for i,tl in enumerate(yticklabels): yticklabels_[convy[i]] = tl
			yticklabels = yticklabels_

	sns.set()
	sns.set_style("ticks")
	plt.figure(figsize=(9*coff, 9*coff))  # in inches

	# count first for better visualization
	nx = max(xindices)+1
	ny = max(yindices)+1
	cnt = np.zeros([nx, ny])
	for x, y in zip(xindices, yindices): cnt[x,y] += 1

	for x in range(nx):
		for y in range(ny):
			idx = [i for i,pos in enumerate(zip(xindices, yindices)) if pos[0]==x and pos[1]==y]
			labels2draw = [labels[i] for i in idx]
			angles = np.linspace(-np.pi/2, np.pi/2, num=len(labels2draw)+1)

			if cnt[x,y] != 0:
				plt.scatter(x,y,s=cnt[x,y]*400*coff*coff,c=sns.xkcd_rgb["dark grey"])

			# if len(labels2draw)<2: r = 0
			# else:                  r = offset

			# for l, a in zip(labels2draw, angles):
			# 	# import pdb; pdb.set_trace()
			# 	x_ = x + np.cos(a)*r
			# 	y_ = y + np.sin(a)*r
			# 	plt.scatter(x_, y_,c=sns.xkcd_rgb["pale red"])
			# 	plt.annotate(l,
			# 		 xy=(x_, y_),
			# 		 xytext=(5, 2),
			# 		 textcoords='offset points',
			# 		 ha='right',
			# 		 va='bottom')

	ax = plt.gca()
	ax.set_xticks(range(nx))
	ax.set_yticks(range(ny))

	fig = plt.gcf()
	fig.subplots_adjust(left=0.25,bottom=0.25)


	if xticklabels is not None:
		xticklabels = [x.lower() for x in xticklabels]
		ax.set_xticklabels(xticklabels, rotation=90, ha='right',size=10*coff)
	else:
		ax.set_xticklabels([""]*nx)
		
	if yticklabels is not None:
		ax.set_yticklabels(yticklabels,size=10*coff)
	else:
		ax.set_yticklabels([""]*ny)

	# plt.grid()
	plt.axes().set_aspect('equal')
	# plt.show()

def plotBubbleOC(xindices, yindices, labels, xticklabels = None, yticklabels = None, offset = 0.25, sort = False):
	new_xindices, new_yindices, new_labels = unfoldOC(xindices,yindices,labels)
	plotBubble(new_xindices, new_yindices, new_labels, xticklabels, yticklabels, offset, sort)


def plotSimpifiedBubble(xindices, yindices, labels, xticklabels = None, yticklabels = None, offset = 0.25, sort = False, xFilteredStr = None, yFilteredStr = None, showLabel = True, team_number = None):
	assert len(xindices) == len(yindices)
	if sort:
		convx, convy = get_sorted_idx(xindices, yindices)
		xindices = [convx[x] for x in xindices]
		yindices = [convy[y] for y in yindices]

		if xticklabels is not None:
			xticklabels_ = xticklabels[:]
			for i,tl in enumerate(xticklabels): xticklabels_[convx[i]] = tl
			xticklabels = xticklabels_
		if yticklabels is not None:
			yticklabels_ = yticklabels[:]
			for i,tl in enumerate(yticklabels): yticklabels_[convy[i]] = tl
			yticklabels = yticklabels_

	sns.set()
	sns.set_style("ticks")
	plt.figure(figsize=(9, 9))  # in inches

	nx = max(xindices)+1
	ny = max(yindices)+1
	cnt = np.zeros([nx, ny])

	xticklabels_ = [""]*len(xticklabels)
	yticklabels_ = [""]*len(yticklabels)

	if xFilteredStr != None:
		try:
			xFilteredIndex = xticklabels.index(xFilteredStr)
		except Exception as e:
			raise e

		for x, y in zip(xindices, yindices): 
			if x == xFilteredIndex:
				cnt[x,y] += 1
				xticklabels_[x] = xticklabels[x]
				yticklabels_[y] = yticklabels[y]
				
				
		xticklabels = xticklabels_
		yticklabels = yticklabels_
		

	if yFilteredStr != None:
		try:
			yFilteredIndex = yticklabels.index(yFilteredStr)
		except Exception as e:
			raise e

		for x, y in zip(xindices, yindices): 
			if y == yFilteredIndex:
				cnt[x,y] += 1
				xticklabels_[x] = xticklabels[x]
				yticklabels_[y] = yticklabels[y]
		xticklabels = xticklabels_
		yticklabels = yticklabels_

	for x in range(nx):
		for y in range(ny):
			# print cnt[x,y]
			if cnt[x,y] != 0.0:
				idx = [i for i,pos in enumerate(zip(xindices, yindices)) if pos[0]==x and pos[1]==y]
				labels2draw = [labels[i] for i in idx]
				angles = np.linspace(-np.pi/2, np.pi/2, num=len(labels2draw)+1)
				dis = range(len(labels2draw))
				plt.scatter(x,y,s=cnt[x,y]*400,c=sns.xkcd_rgb["dark grey"])

				if showLabel:

					if len(labels2draw)<2: r = 0
					else:                  r = offset

					for l, a in zip(labels2draw, dis):
						# import pdb; pdb.set_trace()
						# x_ = x + np.cos(a)*r
						# y_ = y + np.sin(a)*r
						x_ = x + 0.5 + a 
						y_ = y

						# plt.scatter(x_, y_,c=sns.xkcd_rgb["pale red"])
						plt.annotate(l,
							 xy=(x_, y_),
							 xytext=(5, 2),
							 textcoords='offset points',
							 ha='right',
							 va='bottom')
	ax = plt.gca()
	ax.set_xticks(range(0,nx))
	ax.set_yticks(range(0,ny))
	plt.ylim(-1,ny)
	plt.xlim(-1,nx)

	# sns.despine(left=True);
	# sns.despine(offset=50, trim=True);

	fig = plt.gcf()
	fig.subplots_adjust(left=0.2)
	# if team_number != None:
	# 	fig.suptitle('Team ' + str(team_number),fontsize=20)

	if showLabel:
		if xticklabels is not None:
			# ax.set_xticklabels(xticklabels, rotation=20, ha='right')
			ax.set_xticklabels(xticklabels)
		if yticklabels is not None:
			ax.set_yticklabels(yticklabels)
	else:
		ax.set_xticklabels([""]*len(xticklabels))
		ax.set_yticklabels([""]*len(yticklabels))

	# plt.grid()
	# plt.xlabel("Human Clustering")
	# plt.ylabel("Machine Clustering")
	plt.axes().set_aspect('equal')

def plotSimpifiedBubbleOC(xindices, yindices, labels, xticklabels = None, yticklabels = None, offset = 0.25, sort = False, xFilteredStr = None, yFilteredStr = None, showLabel = True, team_number = None):
	new_xindices, new_yindices, new_labels = unfoldOC(xindices,yindices,labels)
	plotSimpifiedBubble(new_xindices,new_yindices,new_labels,xticklabels, yticklabels, offset, sort, xFilteredStr, yFilteredStr, showLabel, team_number)

def unfoldOC(xindices, yindices, labels):
	assert len(xindices)==len(yindices)==len(labels)

	length = len(xindices)

	x_list = list()
	y_list = list()
	label_list = list()
	# print xindices
	# print yindices

	for i in range(length):
		# print xindices[i]
		for x_index in xindices[i]:
			x_list.append(x_index)
			y_list.append(yindices[i])
			label_list.append(labels[i])

	return x_list, y_list, label_list







def get_sorted_idx(xindices, yindices):
	nx = max(xindices)+1
	ny = max(yindices)+1

	cnt = np.zeros([ny, nx])
	for x, y in zip(xindices, yindices):
			cnt[y,x] += 1
	chk = cnt.copy()

	oldx = list(range(nx))
	oldy = list(range(ny))
	newx = []
	newy = []
	while len(oldx)*len(oldy) > 1:
		r, c = np.unravel_index(np.argmax(chk), chk.shape)
		if len(oldx)>1: 
			newx.append(oldx.pop(c))
			chk = np.delete(chk, c, axis=1)
		if len(oldy)>1: 
			newy.append(oldy.pop(r))
			chk = np.delete(chk, r, axis=0)
	newx += oldx
	newy += oldy
	
	converterx = {old:newx.index(old) for old in range(nx)}
	convertery = {old:newy.index(old) for old in range(ny)}
	return converterx, convertery

def main():
	m_labels_index = [ 5,  5,  5,  2,  8,  8,  8, 11,  7, 15, 11,  3, 10, 12, 10,  1, 10,12,  1, 12,  5,  1,  2,  9,  1, 10,  9, 10, 12,  1,  2,  3,  3, 10,10,  2,  4,  4, 10, 10,  6,  8, 10, 10,  8,  3, 10,  8, 10,  2,  2,10,  5,  8,  7, 11, 11, 13, 14, 13,  5,  7,  2,  7, 13, 10,  8,  8,8,  8,  2,  8,  2,  7,  8,  8,  8,  8,  8, 13,  7,  7]
	m_labels_index = [x-1 for x in m_labels_index]
	m_labels = ['system;people', 'system;vehicle', 'control;vehicle', 'use;projection', 'car;control', 'higher price service', 'car;system', 'car;driver', 'car;people', 'vehicle;car', 'pedals;bed', 'display;screen', 'screen;color', 'driving map', 'self parking']
	h_labels_index = [0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 0, 0, 1, 1, 5, 5, 6, 7, 3, 8, 0, 1, 9, 2, 10, 7, 7, 7, 8, 11, 0, 0, 0, 4, 4, 11, 10, 10, 12, 12, 13, 13, 13, 1, 2, 13, 13, 7, 13, 13, 13, 7, 0, 1, 9, 2, 7, 3, 8, 8, 11, 14, 0, 9, 2, 2, 5, 4, 8, 8, 14, 14, 10, 13, 13, 13, 13, 13, 7, 7, 7, 13]
	h_labels = ['ALTERNATIVE CONTROL', 'CAR-TO-CAR CONNECTIVITY', 'COMFORT', 'NAVIGATION', 'PARKING', 'ENTRY & EXIT', 'EXTERNAL SIGNAL', 'LEISURE & ENTERTAINMENT', 'SCREEN & DISPLAYS', 'CLOUD TECHNOLOGY', 'IN-CAR COMMUNICATION', 'VOICE PROGRAM', 'PORTABLE DEVICE CONNECTIVITY', 'SAFETY', 'WARNING SYSTEM']
	concept = range(len(m_labels_index))
	# plotBubble(h_labels_index,m_labels_index,concept,h_labels,m_labels,sort=True)
	plotSimpifiedBubble(h_labels_index,m_labels_index,concept,h_labels,m_labels,xFilteredStr='COMFORT',showLabel=True,team_number=1)
	# plotSimpifiedBubble(h_labels_index,m_labels_index,concept,h_labels,m_labels,yFilteredStr='use;projection',showLabel=True)
	plt.show()

def test():
	m_labels_index = [3, 3, 3, 1, 6, 6, 6, 9, 5, 12, 9, 2, 8, 10, 8, 0, 8, 10, 0, 10, 3, 0, 1, 7, 0, 8, 7, 8, 10, 0, 1, 2, 2, 8, 8, 1, 3, 3, 8, 8, 4, 6, 8, 8, 6, 2, 8, 6, 8, 1, 1, 8, 3, 6, 5, 9, 9, 10, 11, 10, 3, 5, 1, 5, 10, 8, 6, 6, 6, 6, 1, 6, 1, 5, 6, 6, 6, 6, 6, 10, 5, 5]
	h_labels_index = [0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 0, 0, 1, 1, 5, 5, 6, 7, 3, 8, 0, 1, 9, 2, 10, 7, 7, 7, 8, 11, 0, 0, 0, 4, 4, 11, 10, 10, 12, 12, 13, 13, 13, 1, 2, 13, 13, 7, 13, 13, 13, 7, 0, 1, 9, 2, 7, 3, 8, 8, 11, 14, 0, 9, 2, 2, 5, 4, 8, 8, 14, 14, 10, 13, 13, 13, 13, 13, 7, 7, 7, 13]
	concept = range(len(m_labels_index))
	plotBubble(h_labels_index,m_labels_index,concept,sort=True)
	plt.show()


if __name__ == '__main__':
	main()