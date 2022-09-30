# From https://stackabuse.com/python-how-to-flatten-list-of-lists/
import numpy as np

def flatten(element):
	returnList = []
	if isinstance(element, list) == True:
		for anItem in element:
			returnList = returnList + flatten(anItem)
	else:
		returnList = [element]
	return returnList

def repeat(list):
	maxLength = len(list[0])
	for aSubList in list:
		newLength = len(aSubList)
		if newLength > maxLength:
			maxLength = newLength
	for anItem in list:
		if (len(anItem) > 0):
			itemToAppend = anItem[-1]
		else:
			itemToAppend = None
		for i in range(len(anItem), maxLength):
			anItem.append(itemToAppend)
	return list

# From https://stackoverflow.com/questions/34432056/repeat-elements-of-list-between-each-other-until-we-reach-a-certain-length
def onestep(cur,y,base):
    # one step of the iteration
    if cur is not None:
        y.append(cur)
        base.append(cur)
    else:
        y.append(base[0])  # append is simplest, for now
        base = base[1:]+[base[0]]  # rotate
    return base

def iterate(list):
	maxLength = len(list[0])
	returnList = []
	for aSubList in list:
		newLength = len(aSubList)
		if newLength > maxLength:
			maxLength = newLength
	for anItem in list:
		for i in range(len(anItem), maxLength):
			anItem.append(None)
		y=[]
		base=[]
		for cur in anItem:
			base = onestep(cur,y,base)
		returnList.append(y)
	return returnList

def trim(list):
	minLength = len(list[0])
	returnList = []
	for aSubList in list:
		newLength = len(aSubList)
		if newLength < minLength:
			minLength = newLength
	for anItem in list:
		anItem = anItem[:minLength]
		returnList.append(anItem)
	return returnList

# Adapted from https://stackoverflow.com/questions/533905/get-the-cartesian-product-of-a-series-of-lists
def interlace(ar_list):
    if not ar_list:
        yield []
    else:
        for a in ar_list[0]:
            for prod in interlace(ar_list[1:]):
                yield [a,]+prod

def interlace2(ar_list):
    if not ar_list:
        yield []
    else:
        for a in ar_list[0]:
            for prod in interlace(ar_list[1:]):
                yield [a,]+prod

def transposeList(l):
	length = len(l[0])
	returnList = []
	for i in range(length):
		tempRow = []
		for j in range(len(l)):
			tempRow.append(l[j][i])
		returnList.append(tempRow)
	return returnList

def list_level_iter(lst, level, _current_level: int= 1):
    """
    Iterate over all lists with given nesting
    With level 1 it will return the given list
    With level 2 it will iterate over all nested lists in the main one
    If a level does not have lists on that level it will return empty list
    _current_level - for internal use only
    """
    if _current_level < level:
        try:
            for nested_lst in lst:
                if not isinstance(nested_lst, list):
                    raise TypeError
                yield from list_level_iter(nested_lst, level, _current_level + 1)
        except TypeError:
            yield []
    else:
        yield lst

def replicateInputs(inputs, replicationType):
	#assert isinstance(inputs, list), "Replication.replicateInputs: inputs must be a list"
	#assert isinstance(replicationType, str), "Replication.replicateInputs: replicationType must be a string"
	replicationType = replicationType.lower()
	if (replicationType == "default" or replicationType == "iterate"):
		inputs = iterate(inputs)
		inputs = transposeList(inputs)
	elif (replicationType == "trim"):
		inputs = trim(inputs)
		inputs = transposeList(inputs)
	elif (replicationType == "repeat"):
		inputs = repeat(inputs)
		inputs = transposeList(inputs)
	elif (replicationType == "interlace"):
		inputs = list(interlace(inputs))
	else:
		raise Exception("Error - Replication.replicateInputs: Replication type is unrecognized. It must be one of: default, iterate, trim, repeat, or interlace.")
	return inputs

def reassemble(flatList, nestedList, index):
	output = []
	if nestedList == None:
		return []
	if isinstance(nestedList, list):
		for subItem in nestedList:
			result = reassemble(flatList, subItem, index)
			x = result[0]
			index = result[1]
			output.append(x)
	else:
		try:
			output = flatList[index]
		except:
			output = None
		index = index+1
	return [output, index]

def unflatten(flatList, nestedList):
	return reassemble(flatList, nestedList, 0)[0]

replication = [("Default", "Default", "", 1),("Trim", "Trim", "", 2),("Iterate", "Iterate", "", 3),("Repeat", "Repeat", "", 4),("Interlace", "Interlace", "", 5)]

def shortest_longest_list(inputs_nested, inputs_flat):
	lengths = [len(x) for x in inputs_flat]
	indices = list(range(len(lengths)))
	sorted_list = [x for _, x in sorted(zip(lengths, indices))]
	return [inputs_nested[sorted_list[0]], inputs_nested[sorted_list[-1]]]

def best_match(inputs_nested, inputs_flat, replicationType):
	if replicationType == "Trim":
		return shortest_longest_list(inputs_nested, inputs_flat)[0]
	else:
		return shortest_longest_list(inputs_nested, inputs_flat)[1]

def re_interlace(output_list, input_lists):
	dimensions = [len(x) for x in input_lists if len(x) > 1]
	if len(dimensions) > 1:
		a = np.array(output_list)
		return a.reshape(dimensions).tolist()
	else:
		return output_list