'''
For each code file mentioned in 'py150_file/python50k_eval.txt' and 'py150_file/python100k_train.txt',
reads the code file in 'py150_file/data' and the structure information in 'py150_structures',
processes these data (tokenizes the code file and graphizes the structure),
then writes the output in 'py150_new/python50k_eval.txt' and 'py150_new/python100k_train.txt'
'''

import os
import time
import tokenize
import random

def getGraph(path, rootName):
    nodes = []
    edges = []

    # read structure nodes
    file = open(path + 'node.txt')
    context = file.read()
    file.close()
    context = context.split('\n')
    context = context[1 : -1]
    infos = []
    for y in context:
        infos.append([int(y[0 : y.find(',')]), # Id
                      y[y.find(',') + 1 : y.find(',', y.find(',') + 1)], # Type
                      y[y.find(',', y.find(',') + 1) + 1 : y.rfind(',')], # Name
                      int(y[y.rfind(',') + 1 :])]) # Father
        # each infos[i] is a node
        # infos[i][0] is Id
        # infos[i][1] is Type (Package/File/Function/Class/Variable)
        # infos[i][2] is Name
        # infos[i][3] is Father (-1 if no father)

    typeMap = dict()
    rootId = -1
    for i in range(len(infos)):
        typeMap[infos[i][0]] = infos[i][1] # typeMap is a map from Id to Type
        if ((infos[i][1] == 'File') and (infos[i][2][-len(rootName) :] == rootName)):
            rootId = infos[i][0]
    if (rootId == -1): # ENRE fails to extract the root node
        return -1, -1, -1, -1

    # remove the 'Variable' nodes and the non-level-1 'Function'/'Class' nodes
    idMap = dict() # idMap is a map from old id to new id
    rootChild = dict() # rootChild is a map from name of root's direct child to their new id
    n = 0
    for i in range(len(infos)):
        if (infos[i][1] == 'Variable'):
            continue
        if ((infos[i][3] == -1) or (typeMap[infos[i][3]] == 'Package') or (typeMap[infos[i][3]] == 'File')):
            nodes.append(infos[i][2])
            idMap[infos[i][0]] = n
            if (infos[i][3] == rootId):
                assert((infos[i][1] == 'Function') or (infos[i][1] == 'Class'))
                rootChild[infos[i][2]] = n
            n += 1
    rootId = idMap[rootId]

    # add edges: child -> father
    for x in infos:
        if ((x[0] in idMap) and (x[3] in idMap)):
            edges.append([idMap[x[0]], idMap[x[3]]])

    # read structure edges
    file = open(path + 'edge.txt')
    context = file.read()
    file.close()
    context = context.split('\n')
    context = context[1 : -1]
    # add edges: target -> source
    for x in context:
        source = int(x[0 : x.find(',')])
        target = int(x[x.find(',') + 1 : x.find(',', x.find(',') + 1)])
        if ((source in idMap) and (target in idMap)):
            edges.append([idMap[target], idMap[source]])
    return nodes, edges, rootId, rootChild

def cutGraph(nodes, edges, rootId, rootChild):
    # 'Modified BFS' to decide which nodes to remain
    n = len(nodes)
    e = [[] for i in range(n)]
    for x in edges:
        e[x[0]].append(x[1])
        e[x[1]].append(x[0])
    remainNodes = set() # nodes that remain
    remainNodes.add(rootId)
    while (len(remainNodes) < 200):
        adding = []
        for u in remainNodes:
            for v in e[u]:
                if (v not in remainNodes):
                    adding.append(v)
        if (len(adding) == 0):
            break
        random.shuffle(adding)
        for x in adding:
            remainNodes.add(x)
            if (len(remainNodes) == 200):
                break

    # build a new graph base on remainNodes
    newNodes = [] # remained nodes
    newEdges = [] # remained edges
    newChild = dict() # remained root's direct child
    idMap = dict() # a map from old id to new id
    idMap[rootId] = 0 # new id of root must be 0
    m = 1
    newNodes.append(nodes[rootId])
    for x in remainNodes:
        if (x != rootId):
            idMap[x] = m
            m += 1
            newNodes.append(nodes[x])
    for x in edges:
        if ((x[0] in idMap) and (x[1] in idMap)):
            newEdges.append([idMap[x[0]], idMap[x[1]]])
    for (x, y) in rootChild.items():
        if (y in idMap):
            newChild[x] = idMap[y]
    return newNodes, newEdges, newChild

def getCodeTokens(path, rootChild):
    tokens = []
    crossEdges = []
    IndentNumber = 0
    with open(path, 'rb') as f:
        Generator = tokenize.tokenize(f.readline)
        for x in Generator:
            # x is a token from tokenize.tokenize(f.readline)
            # x.type: token's type
            # x.string: token's text
            if (x.type == 0 or x.type == 59 or x.type == 60 or x.type == 62): # COMMENT
                pass
            elif (x.type == 4 or x.type == 61): # NEWLINE
                pass
            elif (x.type == 5): # INDENT
                tokens.append('5INDENT') # special token for indent
                IndentNumber += 1
            elif (x.type == 6): # DEDENT
                tokens.append('6DEDENT') # special token for dedent
                IndentNumber -= 1
            elif (x.type == 1 or x.type == 2 or x.type == 54): # NAME NUMBER OP
                if ((IndentNumber == 0) and (len(tokens) != 0) and ((tokens[-1] == 'def') or (tokens[-1] == 'class'))):
                    if (x.string in rootChild):
                        # add cross edges: graph node -> token
                        crossEdges.append([rootChild[x.string], len(tokens)])
                tokens.append(x.string)
            elif (x.type == 3): # STRING
                tokens.append(x.string)
            else:
                assert(False)
    return tokens, crossEdges

def work(datasetName):
    startTime = time.time()
    file = open('py150_files/' + datasetName)
    context = file.read()
    file.close()
    fileList = context[0 : -1].split('\n')
    
    cnt = 0
    cntFail = 0
    file = open('py150_new/' + datasetName, 'w') # output file
    for x in fileList: # x is each code file name
        cnt += 1
        dir1 = x[5 : x.find('/', 5)]
        dir2 = x[x.find('/', 5) + 1 : x.find('/', x.find('/', 5) + 1)]
        # dir1/dir2 is the name of project the code file belongs to
        assert os.path.isdir('py150_files/data/' + dir1 + '/' + dir2)
        assert os.path.isfile('py150_files/' + x)
        if (not os.path.isdir('py150_structures/' + dir1 + '/' + dir2)): # ENRE fails on the project
            cntFail += 1
            continue
        assert(os.path.isfile('py150_structures/' + dir1 + '/' + dir2 + '/node.txt'))
        assert(os.path.isfile('py150_structures/' + dir1 + '/' + dir2 + '/edge.txt'))

        nodes, edges, rootId, rootChild = getGraph('py150_structures/' + dir1 + '/' + dir2 + '/', x) # read the graph
        # nodes, edges: the graph that ENRE outputs
        # rootId: id of x (we call x as root)
        # rootChild: a dict that map name of root's direct child to id
        if (rootId == -1): # ENRE fails to extract the root node
            cntFail += 1
            continue
        nodes, edges, rootChild = cutGraph(nodes, edges, rootId, rootChild) # cut the graph
        tokens, crossEdges = getCodeTokens('py150_files/' + x, rootChild) # read the tokens
        # tokens: the tokenized code file
        # crossEdges: edges between graph nodes and token nodes
        info = dict()
        info['path'] = x
        info['tokens'] = tokens
        info['nodes'] = nodes
        info['edges'] = edges
        info['crossEdges'] = crossEdges
        print(info, file = file) # write to output file
        if ((cnt - cntFail) % 1 == 0): # print using time
            print('done', cnt - cntFail, 'using time ', round(time.time() - startTime, 2))
    file.close()
    print(cnt - cntFail, 'in total')

os.system('mkdir py150_new')
work('python50k_eval.txt')
work('python100k_train.txt')
