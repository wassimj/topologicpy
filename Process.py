import topologic

def internalVertex(topology, tolerance):
	vst = None
	classType = topology.Type()
	if classType == 64: #CellComplex
		tempCells = []
		_ = topology.Cells(None,tempCells)
		tempCell = tempCells.front()
		vst = topologic.CellUtility.InternalVertex(tempCell, tolerance)
	elif classType == 32: #Cell
		vst = topologic.CellUtility.InternalVertex(topology, tolerance)
	elif classType == 16: #Shell
		tempFaces = []
		_ = topology.Faces(None, tempFaces)
		tempFace = tempFaces.front()
		vst = topologic.FaceUtility.InternalVertex(tempFace, tolerance)
	elif classType == 8: #Face
		vst = topologic.FaceUtility.InternalVertex(topology, tolerance)
	elif classType == 4: #Wire
		if topology.IsClosed():
			internalBoundaries = []
			tempFace = topologic.Face.ByExternalInternalBoundaries(topology, internalBoundaries)
			vst = topologic.FaceUtility.InternalVertex(tempFace, tolerance)
		else:
			tempEdges = []
			_ = topology.Edges(None, tempEdges)
			vst = topologic.EdgeUtility.PointAtParameter(tempVertex.front(), 0.5)
	elif classType == 2: #Edge
		vst = topologic.EdgeUtility.PointAtParameter(topology, 0.5)
	elif classType == 1: #Vertex
		vst = topology
	else:
		vst = topology.CenterOfMass()
	return vst

def listAttributeValues(listAttribute):
	listAttributes = listAttribute.ListValue()
	returnList = []
	for attr in listAttributes:
		if isinstance(attr, topologic.IntAttribute):
			returnList.append(attr.IntValue())
		elif isinstance(attr, topologic.DoubleAttribute):
			returnList.append(attr.DoubleValue())
		elif isinstance(attr, topologic.StringAttribute):
			returnList.append(attr.StringValue())
	return returnList

def getValues(item):
	keys = item.Keys()
	returnList = []
	for key in keys:
		try:
			attr = item.ValueAtKey(key)
		except:
			raise Exception("Dictionary.Values - Error: Could not retrieve a Value at the specified key ("+key+")")
		if isinstance(attr, topologic.IntAttribute):
			returnList.append(attr.IntValue())
		elif isinstance(attr, topologic.DoubleAttribute):
			returnList.append(attr.DoubleValue())
		elif isinstance(attr, topologic.StringAttribute):
			returnList.append(attr.StringValue())
		elif isinstance(attr, topologic.ListAttribute):
			returnList.append(listAttributeValues(attr))
		else:
			returnList.append("")
	return returnList

def getKeys(item):
	stlKeys = item.Keys()
	returnList = []
	copyKeys = stlKeys.__class__(stlKeys) #wlav suggested workaround. Make a copy first
	for x in copyKeys:
		k = x.c_str()
		returnList.append(k)
	return returnList

def getValueAtKey(item, key):
	try:
		attr = item.ValueAtKey(key)
	except:
		raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
	if isinstance(attr, topologic.IntAttribute):
		return (attr.IntValue())
	elif isinstance(attr, topologic.DoubleAttribute):
		return (attr.DoubleValue())
	elif isinstance(attr, topologic.StringAttribute):
		return (attr.StringValue())
	elif isinstance(attr, topologic.ListAttribute):
		return (listAttributeValues(attr))
	else:
		return None

def processKeysValues(keys, values):
	if len(keys) != len(values):
		raise Exception("DictionaryByKeysValues - Keys and Values do not have the same length")
	stl_keys = []
	stl_values = []
	for i in range(len(keys)):
		if isinstance(keys[i], str):
			stl_keys.append(keys[i])
		else:
			stl_keys.append(str(keys[i]))
		if isinstance(values[i], list) and len(values[i]) == 1:
			value = values[i][0]
		else:
			value = values[i]
		if isinstance(value, bool):
			if value == False:
				stl_values.append(topologic.IntAttribute(0))
			else:
				stl_values.append(topologic.IntAttribute(1))
		elif isinstance(value, int):
			stl_values.append(topologic.IntAttribute(value))
		elif isinstance(value, float):
			stl_values.append(topologic.DoubleAttribute(value))
		elif isinstance(value, str):
			stl_values.append(topologic.StringAttribute(value))
		elif isinstance(value, list):
			l = []
			for v in value:
				if isinstance(v, bool):
					l.append(topologic.IntAttribute(v))
				elif isinstance(v, int):
					l.append(topologic.IntAttribute(v))
				elif isinstance(v, float):
					l.append(topologic.DoubleAttribute(v))
				elif isinstance(v, str):
					l.append(topologic.StringAttribute(v))
			stl_values.append(topologic.ListAttribute(l))
		else:
			raise Exception("Error: Value type is not supported. Supported types are: Boolean, Integer, Double, String, or List.")
	myDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
	return myDict

def mergeDictionaries(sources):
	if isinstance(sources, list) == False:
		sources = [sources]
	sinkKeys = []
	sinkValues = []
	d = sources[0].GetDictionary()
	if d != None:
		stlKeys = d.Keys()
		if len(stlKeys) > 0:
			sinkKeys = d.Keys()
			sinkValues = getValues(d)
	for i in range(1,len(sources)):
		d = sources[i].GetDictionary()
		if d == None:
			continue
		stlKeys = d.Keys()
		if len(stlKeys) > 0:
			sourceKeys = d.Keys()
			for aSourceKey in sourceKeys:
				if aSourceKey not in sinkKeys:
					sinkKeys.append(aSourceKey)
					sinkValues.append("")
			for i in range(len(sourceKeys)):
				index = sinkKeys.index(sourceKeys[i])
				sourceValue = getValueAtKey(d, sourceKeys[i])
				if sourceValue != None:
					if sinkValues[index] != "":
						if isinstance(sinkValues[index], list):
							sinkValues[index].append(sourceValue)
						else:
							sinkValues[index] = [sinkValues[index], sourceValue]
					else:
						sinkValues[index] = sourceValue
	if len(sinkKeys) > 0 and len(sinkValues) > 0:
		return processKeysValues(sinkKeys, sinkValues)
	return None

def mergeDictionaries2(sources):
	if isinstance(sources, list) == False:
		sources = [sources]
	sinkKeys = []
	sinkValues = []
	d = sources[0]
	if d != None:
		stlKeys = d.Keys()
		if len(stlKeys) > 0:
			sinkKeys = d.Keys()
			sinkValues = getValues(d)
	for i in range(1,len(sources)):
		d = sources[i]
		if d == None:
			continue
		stlKeys = d.Keys()
		if len(stlKeys) > 0:
			sourceKeys = d.Keys()
			for aSourceKey in sourceKeys:
				if aSourceKey not in sinkKeys:
					sinkKeys.append(aSourceKey)
					sinkValues.append("")
			for i in range(len(sourceKeys)):
				index = sinkKeys.index(sourceKeys[i])
				sourceValue = getValueAtKey(d, sourceKeys[i])
				if sourceValue != None:
					if sinkValues[index] != "":
						if isinstance(sinkValues[index], list):
							sinkValues[index].append(sourceValue)
						else:
							sinkValues[index] = [sinkValues[index], sourceValue]
					else:
						sinkValues[index] = sourceValue
	if len(sinkKeys) > 0 and len(sinkValues) > 0:
		return processKeysValues(sinkKeys, sinkValues)
	return None

def processCellComplex(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	graph = None
	edges = []
	vertices = []
	cellmat = []
	if direct == True:
		cells = []
		_ = topology.Cells(None, cells)
		# Create a matrix of zeroes
		for i in range(len(cells)):
			cellRow = []
			for j in range(len(cells)):
				cellRow.append(0)
			cellmat.append(cellRow)
		for i in range(len(cells)):
			for j in range(len(cells)):
				if (i != j) and cellmat[i][j] == 0:
					cellmat[i][j] = 1
					cellmat[j][i] = 1
					sharedt = []
					cells[i].SharedTopologies(cells[j], 8, sharedt)
					if len(sharedt) > 0:
						if useInternalVertex == True:
							v1 = topologic.CellUtility.InternalVertex(cells[i], tolerance)
							v2 = topologic.CellUtility.InternalVertex(cells[j], tolerance)
						else:
							v1 = cells[i].CenterOfMass()
							v2 = cells[j].CenterOfMass()
						e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
						mDict = mergeDictionaries(sharedt)
						if mDict:
							e.SetDictionary(mDict)
						edges.append(e)
	if directApertures == True:
		cellmat = []
		cells = []
		_ = topology.Cells(None, cells)
		# Create a matrix of zeroes
		for i in range(len(cells)):
			cellRow = []
			for j in range(len(cells)):
				cellRow.append(0)
			cellmat.append(cellRow)
		for i in range(len(cells)):
			for j in range(len(cells)):
				if (i != j) and cellmat[i][j] == 0:
					cellmat[i][j] = 1
					cellmat[j][i] = 1
					sharedt = []
					cells[i].SharedTopologies(cells[j], 8, sharedt)
					if len(sharedt) > 0:
						apertureExists = False
						for x in sharedt:
							apList = []
							_ = x.Apertures(apList)
							if len(apList) > 0:
								apTopList = []
								for ap in apList:
									apTopList.append(ap.Topology())
								apertureExists = True
								break
						if apertureExists:
							if useInternalVertex == True:
								v1 = topologic.CellUtility.InternalVertex(cells[i], tolerance)
								v2 = topologic.CellUtility.InternalVertex(cells[j], tolerance)
							else:
								v1 = cells[i].CenterOfMass()
								v2 = cells[j].CenterOfMass()
							e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
							mDict = mergeDictionaries(apTopList)
							if mDict:
								e.SetDictionary(mDict)
							edges.append(e)

	cells = []
	_ = topology.Cells(None, cells)
	if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		for aCell in cells:
			if useInternalVertex == True:
				vCell = topologic.CellUtility.InternalVertex(aCell, tolerance)
			else:
				vCell = aCell.CenterOfMass()
			d1 = aCell.GetDictionary()
			if storeBRep:
				d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [aCell.String(), aCell.Type(), aCell.GetTypeAsString()])
				d3 = mergeDictionaries2([d1, d2])
				_ = vCell.SetDictionary(d3)
			else:
				_ = vCell.SetDictionary(d1)
			vertices.append(vCell)
			faces = []
			_ = aCell.Faces(None, faces)
			sharedTopologies = []
			exteriorTopologies = []
			sharedApertures = []
			exteriorApertures = []
			contents = []
			_ = aCell.Contents(contents)
			for aFace in faces:
				cells = []
				_ = aFace.Cells(topology, cells)
				if len(cells) > 1:
					sharedTopologies.append(aFace)
					apertures = []
					_ = aFace.Apertures(apertures)
					for anAperture in apertures:
						sharedApertures.append(anAperture)
				else:
					exteriorTopologies.append(aFace)
					apertures = []
					_ = aFace.Apertures(apertures)
					for anAperture in apertures:
						exteriorApertures.append(anAperture)
			if viaSharedTopologies:
				for sharedTopology in sharedTopologies:
					if useInternalVertex == True:
						vst = internalVertex(sharedTopology, tolerance)
					else:
						vst = sharedTopology.CenterOfMass()
					d1 = sharedTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedTopology.String(), sharedTopology.Type(), sharedTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = sharedTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							d1 = content.GetDictionary()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X(), vst2.Y(), vst2.Z())
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if viaSharedApertures:
				for sharedAperture in sharedApertures:
					if useInternalVertex == True:
						vst = internalVertex(sharedAperture.Topology(), tolerance)
					else:
						vst = sharedAperture.Topology().CenterOfMass()
					d1 = sharedAperture.Topology().GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedAperture.Topology().String(), sharedAperture.Topology().Type(), sharedAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					if useInternalVertex == True:
						vst = internalVertex(exteriorTopology, tolerance)
					else:
						vst = exteriorTopology.CenterOfMass()
					_ = vst.SetDictionary(exteriorTopology.GetDictionary())
					d1 = exteriorTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorTopology.String(), exteriorTopology.Type(), exteriorTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = exteriorTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							d1 = content.GetDictionary()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = exteriorAperture.Topology().CenterOfMass()
					d1 = exteriorAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorAperture.String(), exteriorAperture.Type(), exteriorAperture.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = aCell.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					d1 = content.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)

	for aCell in cells:
		if useInternalVertex == True:
			vCell = topologic.CellUtility.InternalVertex(aCell, tolerance)
		else:
			vCell = aCell.CenterOfMass()
		d1 = aCell.GetDictionary()
		if storeBRep:
			d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [aCell.String(), aCell.Type(), aCell.GetTypeAsString()])
			d3 = mergeDictionaries2([d1, d2])
			_ = vCell.SetDictionary(d3)
		else:
			_ = vCell.SetDictionary(d1)
		vertices.append(vCell)
	return topologic.Graph.ByVerticesEdges(vertices,edges)

def processCell(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	vertices = []
	edges = []

	if useInternalVertex == True:
		vCell = topologic.CellUtility.InternalVertex(topology, tolerance)
	else:
		vCell = topology.CenterOfMass()
	d1 = topology.GetDictionary()
	if storeBRep:
		d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [topology.String(), topology.Type(), topology.GetTypeAsString()])
		d3 = mergeDictionaries2([d1, d2])
		_ = vCell.SetDictionary(d3)
	else:
		_ = vCell.SetDictionary(d1)
	vertices.append(vCell)

	if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		faces = []
		_ = topology.Faces(None, faces)
		exteriorTopologies = []
		exteriorApertures = []
		for aFace in faces:
			exteriorTopologies.append(aFace)
			apertures = []
			_ = aFace.Apertures(apertures)
			for anAperture in apertures:
				exteriorApertures.append(anAperture)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					if useInternalVertex == True:
						vst = internalVertex(exteriorTopology, tolerance)
					else:
						vst = exteriorTopology.CenterOfMass()
					d1 = exteriorTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorTopology.String(), exteriorTopology.Type(), exteriorTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = exteriorTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = exteriorAperture.Topology().CenterOfMass()
					d1 = exteriorAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorAperture.Topology().String(), exteriorAperture.Topology().Type(), exteriorAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = topology.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					d1 = content.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)

	return topologic.Graph.ByVerticesEdges(vertices, edges)

def processShell(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	graph = None
	edges = []
	vertices = []
	facemat = []
	if direct == True:
		topFaces = []
		_ = topology.Faces(None, topFaces)
		# Create a matrix of zeroes
		for i in range(len(topFaces)):
			faceRow = []
			for j in range(len(topFaces)):
				faceRow.append(0)
			facemat.append(faceRow)
		for i in range(len(topFaces)):
			for j in range(len(topFaces)):
				if (i != j) and facemat[i][j] == 0:
					facemat[i][j] = 1
					facemat[j][i] = 1
					sharedt = []
					topFaces[i].SharedTopologies(topFaces[j], 2, sharedt)
					if len(sharedt) > 0:
						if useInternalVertex == True:
							v1 = topologic.FaceUtility.InternalVertex(topFaces[i], tolerance)
							v2 = topologic.FaceUtility.InternalVertex(topFaces[j], tolerance)
						else:
							v1 = topFaces[i].CenterOfMass()
							v2 = topFaces[j].CenterOfMass()
						e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
						mDict = mergeDictionaries(sharedt)
						if mDict:
							e.SetDictionary(mDict)
						edges.append(e)
	if directApertures == True:
		facemat = []
		topFaces = []
		_ = topology.Faces(None, topFaces)
		# Create a matrix of zeroes
		for i in range(len(topFaces)):
			faceRow = []
			for j in range(len(topFaces)):
				faceRow.append(0)
			facemat.append(faceRow)
		for i in range(len(topFaces)):
			for j in range(len(topFaces)):
				if (i != j) and facemat[i][j] == 0:
					facemat[i][j] = 1
					facemat[j][i] = 1
					sharedt = []
					topFaces[i].SharedTopologies(topFaces[j], 2, sharedt)
					if len(sharedt) > 0:
						apertureExists = False
						for x in sharedt:
							apList = []
							_ = x.Apertures(apList)
							if len(apList) > 0:
								apertureExists = True
								break
						if apertureExists:
							apTopList = []
							for ap in apList:
								apTopList.append(ap.Topology())
							if useInternalVertex == True:
								v1 = topologic.FaceUtility.InternalVertex(topFaces[i], tolerance)
								v2 = topologic.FaceUtility.InternalVertex(topFaces[j], tolerance)
							else:
								v1 = topFaces[i].CenterOfMass()
								v2 = topFaces[j].CenterOfMass()
							e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
							mDict = mergeDictionaries(apTopList)
							if mDict:
								e.SetDictionary(mDict)
							edges.append(e)

	topFaces = []
	_ = topology.Faces(None, topFaces)
	if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		for aFace in topFaces:
			if useInternalVertex == True:
				vFace = topologic.FaceUtility.InternalVertex(aFace, tolerance)
			else:
				vFace = aFace.CenterOfMass()
			_ = vFace.SetDictionary(aFace.GetDictionary())
			vertices.append(vFace)
			fEdges = []
			_ = aFace.Edges(None, fEdges)
			sharedTopologies = []
			exteriorTopologies = []
			sharedApertures = []
			exteriorApertures = []
			for anEdge in fEdges:
				faces = []
				_ = anEdge.Faces(topology, faces)
				if len(faces) > 1:
					sharedTopologies.append(anEdge)
					apertures = []
					_ = anEdge.Apertures(apertures)
					for anAperture in apertures:
						sharedApertures.append(anAperture)
				else:
					exteriorTopologies.append(anEdge)
					apertures = []
					_ = anEdge.Apertures(apertures)
					for anAperture in apertures:
						exteriorApertures.append(anAperture)
			if viaSharedTopologies:
				for sharedTopology in sharedTopologies:
					if useInternalVertex == True:
						vst = internalVertex(sharedTopology, tolerance)
					else:
						vst = sharedTopology.CenterOfMass()
					d1 = sharedTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedTopology.String(), sharedTopology.Type(), sharedTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = sharedTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if viaSharedApertures:
				for sharedAperture in sharedApertures:
					if useInternalVertex == True:
						vst = internalVertex(sharedAperture.Topology(), tolerance)
					else:
						vst = sharedAperture.Topology().CenterOfMass()
					d1 = sharedAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedAperture.Topology().String(), sharedAperture.Topology().Type(), sharedAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					if useInternalVertex == True:
						vst = internalVertex(exteriorTopology, tolerance)
					else:
						vst = exteriorTopology.CenterOfMass()
					d1 = exteriorTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorTopology.String(), exteriorTopology.Type(), exteriorTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = exteriorTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = exteriorAperture.Topology().CenterOfMass()
					d1 = exteriorAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorAperture.Topology().String(), exteriorAperture.Topology().Type(), exteriorAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = aFace.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					d1 = content.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)

	for aFace in topFaces:
		if useInternalVertex == True:
			vFace = internalVertex(aFace, tolerance)
		else:
			vFace = aFace.CenterOfMass()
		d1 = aFace.GetDictionary()
		if storeBRep:
			d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [aFace.String(), aFace.Type(), aFace.GetTypeAsString()])
			d3 = mergeDictionaries2([d1, d2])
			_ = vFace.SetDictionary(d3)
		else:
			_ = vFace.SetDictionary(d1)
		vertices.append(vFace)
	return topologic.Graph.ByVerticesEdges(vertices, edges)

def processFace(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item

	graph = None
	vertices = []
	edges = []

	if useInternalVertex == True:
		vFace = topologic.FaceUtility.InternalVertex(topology, tolerance)
	else:
		vFace = topology.CenterOfMass()
	d1 = topology.GetDictionary()
	if storeBRep:
		d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [topology.String(), topology.Type(), topology.GetTypeAsString()])
		d3 = mergeDictionaries2([d1, d2])
		_ = vFace.SetDictionary(d3)
	else:
		_ = vFace.SetDictionary(d1)
	vertices.append(vFace)
	if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		fEdges = []
		_ = topology.Edges(None, fEdges)
		exteriorTopologies = []
		exteriorApertures = []

		for anEdge in fEdges:
			exteriorTopologies.append(anEdge)
			apertures = []
			_ = anEdge.Apertures(apertures)
			for anAperture in apertures:
				exteriorApertures.append(anAperture)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					if useInternalVertex == True:
						vst = internalVertex(exteriorTopology, tolerance)
					else:
						vst = exteriorTopology.CenterOfMass()
					d1 = exteriorTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorTopology.String(), exteriorTopology.Type(), exteriorTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = exteriorTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = exteriorAperture.Topology().CenterOfMass()
					d1 = exteriorAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorAperture.Topology().String(), exteriorAperture.Topology().Type(), exteriorAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = topology.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					d1 = content.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
	return topologic.Graph.ByVertices(vertices, edges)

def processWire(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	graph = None
	edges = []
	vertices = []
	edgemat = []
	if direct == True:
		topEdges = []
		_ = topology.Edges(None, topEdges)
		# Create a matrix of zeroes
		for i in range(len(topEdges)):
			edgeRow = []
			for j in range(len(topEdges)):
				edgeRow.append(0)
			edgemat.append(edgeRow)
		for i in range(len(topEdges)):
			for j in range(len(topEdges)):
				if (i != j) and edgemat[i][j] == 0:
					edgemat[i][j] = 1
					edgemat[j][i] = 1
					sharedt = []
					topEdges[i].SharedTopologies(topEdges[j], 1, sharedt)
					if len(sharedt) > 0:
						try:
							v1 = topologic.EdgeUtility.PointAtParameter(topEdges[i], 0.5)
						except:
							v1 = topEdges[j].CenterOfMass()
						try:
							v2 = topologic.EdgeUtility.PointAtParameter(topEdges[j], 0.5)
						except:
							v2 = topEdges[j].CenterOfMass()
						e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
						mDict = mergeDictionaries(sharedt)
						if mDict:
							e.SetDictionary(mDict)
						edges.append(e)
	if directApertures == True:
		edgemat = []
		topEdges = []
		_ = topology.Edges(None, topEdges)
		# Create a matrix of zeroes
		for i in range(len(topEdges)):
			edgeRow = []
			for j in range(len(topEdges)):
				edgeRow.append(0)
			edgemat.append(edgeRow)
		for i in range(len(topEdges)):
			for j in range(len(topEdges)):
				if (i != j) and edgemat[i][j] == 0:
					edgemat[i][j] = 1
					edgemat[j][i] = 1
					sharedt = []
					topEdges[i].SharedTopologies(topEdges[j], 1, sharedt)
					if len(sharedt) > 0:
						apertureExists = False
						for x in sharedt:
							apList = []
							_ = x.Apertures(apList)
							if len(apList) > 0:
								apertureExists = True
								break
						if apertureExists:
							try:
								v1 = topologic.EdgeUtility.PointAtParameter(topEdges[i], 0.5)
							except:
								v1 = topEdges[j].CenterOfMass()
							try:
								v2 = topologic.EdgeUtility.PointAtParameter(topEdges[j], 0.5)
							except:
								v2 = topEdges[j].CenterOfMass()
							e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
							apTopologies = []
							for ap in apList:
								apTopologies.append(ap.Topology())
							mDict = mergeDictionaries(apTopologies)
							if mDict:
								e.SetDictionary(mDict)
							edges.append(e)

	topEdges = []
	_ = topology.Edges(None, topEdges)
	if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		for anEdge in topEdges:
			try:
				vEdge = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
			except:
				vEdge = anEdge.CenterOfMass()
			d1 = anEdge.GetDictionary()
			if storeBRep:
				d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [anEdge.String(), anEdge.Type(), anEdge.GetTypeAsString()])
				d3 = mergeDictionaries2([d1, d2])
				_ = vEdge.SetDictionary(d3)
			else:
				_ = vEdge.SetDictionary(d1)
			vertices.append(vEdge)
			eVertices = []
			_ = anEdge.Vertices(None, eVertices)
			sharedTopologies = []
			exteriorTopologies = []
			sharedApertures = []
			exteriorApertures = []
			contents = []
			_ = anEdge.Contents(contents)
			for aVertex in eVertices:
				tempEdges = []
				_ = aVertex.Edges(topology, tempEdges)
				if len(tempEdges) > 1:
					sharedTopologies.append(aVertex)
					apertures = []
					_ = aVertex.Apertures(apertures)
					for anAperture in apertures:
						sharedApertures.append(anAperture)
				else:
					exteriorTopologies.append(aVertex)
					apertures = []
					_ = aVertex.Apertures(apertures)
					for anAperture in apertures:
						exteriorApertures.append(anAperture)
			if viaSharedTopologies:
				for sharedTopology in sharedTopologies:
					vst = sharedTopology.CenterOfMass()
					d1 = sharedTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedTopology.String(), sharedTopology.Type(), sharedTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = sharedTopology.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if viaSharedApertures:
				for sharedAperture in sharedApertures:
					if useInternalVertex == True:
						vst = internalVertex(sharedAperture.Topology(), tolerance)
					else:
						vst = sharedAperture.Topology().CenterOfMass()
					d1 = sharedAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [sharedAperture.Topology().String(), sharedAperture.Topology().Type(), sharedAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["Via Shared Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					vst = exteriorTopology
					vertices.append(exteriorTopology)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = vst.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = extTop.CenterOfMass()
					d1 = extTop.GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [extTop.String(), extTop.Type(), extTop.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = anEdge.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					d1 = content.GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X(), vst.Y(), vst.Z())
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
	for anEdge in topEdges:
		try:
			vEdge = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
		except:
			vEdge = anEdge.CenterOfMass()
		d1 = anEdge.GetDictionary()
		if storeBRep:
			d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [anEdge.String(), anEdge.Type(), anEdge.GetTypeAsString()])
			d3 = mergeDictionaries2([d1, d2])
			_ = vEdge.SetDictionary(d3)
		else:
			_ = vEdge.SetDictionary(d1)
		vertices.append(vEdge)
	return topologic.Graph.ByVerticesEdges(vertices, edges)

def processEdge(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	graph = None
	vertices = []
	edges = []

	if useInternalVertex == True:
		try:
			vEdge = topologic.EdgeUtility.PointAtParameter(topology, 0.5)
		except:
			vEdge = topology.CenterOfMass()
	else:
		vEdge = topology.CenterOfMass()

	d1 = vEdge.GetDictionary()
	if storeBRep:
		d2 = processKeysValues(["brep"], [topology.String()])
		d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [topology.String(), topology.Type(), topology.GetTypeAsString()])
		d3 = mergeDictionaries2([d1, d2])
		_ = vEdge.SetDictionary(d3)
	else:
		_ = vEdge.SetDictionary(topology.GetDictionary())

	vertices.append(vEdge)

	if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
		eVertices = []
		_ = topology.Vertices(None, eVertices)
		exteriorTopologies = []
		exteriorApertures = []
		for aVertex in eVertices:
			exteriorTopologies.append(aVertex)
			apertures = []
			_ = aVertex.Apertures(apertures)
			for anAperture in apertures:
				exteriorApertures.append(anAperture)
			if toExteriorTopologies:
				for exteriorTopology in exteriorTopologies:
					if useInternalVertex == True:
						vst = internalVertex(exteriorTopology, tolerance)
					else:
						vst = exteriorTopology.CenterOfMass()
					d1 = exteriorTopology.GetDictionary()
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorTopology.String(), exteriorTopology.Type(), exteriorTopology.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Topologies"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
					if toContents:
						contents = []
						_ = vst.Contents(contents)
						for content in contents:
							if useInternalVertex == True:
								vst2 = internalVertex(content, tolerance)
							else:
								vst2 = content.CenterOfMass()
							vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
							d1 = content.GetDictionary()
							if storeBRep:
								d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
								d3 = mergeDictionaries2([d1, d2])
								_ = vst2.SetDictionary(d3)
							else:
								_ = vst2.SetDictionary(d1)
							vertices.append(vst2)
							tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
							tempd = processKeysValues(["relationship"],["To Contents"])
							_ = tempe.SetDictionary(tempd)
							edges.append(tempe)
			if toExteriorApertures:
				for exteriorAperture in exteriorApertures:
					extTop = exteriorAperture.Topology()
					if useInternalVertex == True:
						vst = internalVertex(extTop, tolerance)
					else:
						vst = exteriorAperture.Topology().CenterOfMass()
					d1 = exteriorAperture.Topology().GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [exteriorAperture.Topology().String(), exteriorAperture.Topology().Type(), exteriorAperture.Topology().GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					_ = vst.SetDictionary(exteriorAperture.Topology().GetDictionary())
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Exterior Apertures"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
			if toContents:
				contents = []
				_ = topology.Contents(contents)
				for content in contents:
					if useInternalVertex == True:
						vst = internalVertex(content, tolerance)
					else:
						vst = content.CenterOfMass()
					d1 = content.GetDictionary()
					vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
					if storeBRep:
						d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
						d3 = mergeDictionaries2([d1, d2])
						_ = vst.SetDictionary(d3)
					else:
						_ = vst.SetDictionary(d1)
					vertices.append(vst)
					tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
					tempd = processKeysValues(["relationship"],["To Contents"])
					_ = tempe.SetDictionary(tempd)
					edges.append(tempe)
	graph = topologic.Graph.ByVerticesEdges(vertices, edges)
	return graph

def processVertex(item):
	topology, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, useInternalVertex, storeBRep, tolerance = item
	vertices = [topology]
	edges = []
	if toContents:
		contents = []
		_ = topology.Contents(contents)
		for content in contents:
			if useInternalVertex == True:
				vst = internalVertex(content, tolerance)
			else:
				vst = content.CenterOfMass()
			d1 = content.GetDictionary()
			vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
			if storeBRep:
				d2 = processKeysValues(["brep", "brepType", "brepTypeString"], [content.String(), content.Type(), content.GetTypeAsString()])
				d3 = mergeDictionaries2([d1, d2])
				_ = vst.SetDictionary(d3)
			else:
				_ = vst.SetDictionary(d1)
			vertices.append(vst)
			tempe = topologic.Edge.ByStartVertexEndVertex(topology, vst)
			tempd = processKeysValues(["relationship"],["To Contents"])
			_ = tempe.SetDictionary(tempd)
			edges.append(tempe)
	return topologic.Graph.VerticesEdges(vertices, edges)