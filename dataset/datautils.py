from os import path
def datapath(classname,teamno):
	return path.dirname(path.abspath(__file__)) + "/" + classname + "/Team" + str(teamno) + ".csv"

if __name__ == '__main__':
	print datapath("DesInv",2)
	