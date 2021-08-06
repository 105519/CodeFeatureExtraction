import os
import time
import tokenize
import random

def getGraph(path, rootName):
    nodes = []
    edges = []

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

    typeMap = dict()
    rootId = -1
    for i in range(len(infos)):
        typeMap[infos[i][0]] = infos[i][1]
        if ((infos[i][1] == 'File') and (infos[i][2][-len(rootName) :] == rootName)):
            rootId = infos[i][0]
    if (rootId == -1):
        return -1, -1, -1, -1

    idMap = dict()
    rootChild = dict()
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

    # child -> father
    for x in infos:
        if ((x[0] in idMap) and (x[3] in idMap)):
            edges.append([idMap[x[0]], idMap[x[3]]])

    file = open(path + 'edge.txt')
    context = file.read()
    file.close()
    context = context.split('\n')
    context = context[1 : -1]

    # target -> source
    for x in context:
        source = int(x[0 : x.find(',')])
        target = int(x[x.find(',') + 1 : x.find(',', x.find(',') + 1)])
        if ((source in idMap) and (target in idMap)):
            edges.append([idMap[target], idMap[source]])
    return nodes, edges, rootId, rootChild

def cutGraph(nodes, edges, rootId, rootChild):
    n = len(nodes)
    e = [[] for i in range(n)]
    for x in edges:
        e[x[0]].append(x[1])
        e[x[1]].append(x[0])

    remainNodes = set()
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

    newNodes = []
    newEdges = []
    newChild = dict()

    idMap = dict()
    idMap[rootId] = 0
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
        tokenGenerator = tokenize.tokenize(f.readline)
        for x in tokenGenerator:
            if (x.type == 0 or x.type == 59 or x.type == 60 or x.type == 62): # COMMENT
                pass
            elif (x.type == 4 or x.type == 61): # NEWLINE
                pass
            elif (x.type == 5): # INDENT
                tokens.append('5INDENT')
                IndentNumber += 1
            elif (x.type == 6): # DEDENT
                tokens.append('6DEDENT')
                IndentNumber -= 1
            elif (x.type == 1 or x.type == 2 or x.type == 54): # NAME NUMBER OP
                if ((IndentNumber == 0) and (len(tokens) != 0) and ((tokens[-1] == 'def') or (tokens[-1] == 'class'))):
                    if (x.string in rootChild):
                        # graph node -> token
                        crossEdges.append([rootChild[x.string], len(tokens)])
                tokens.append(x.string)
            elif (x.type == 3): # STRING
                tokens.append(x.string)
            else:
                assert(False)
    return tokens, crossEdges

def work(datasetName):
    startTime = time.time()
    file = open('py150_files/' + datasetName + '.txt')
    context = file.read()
    file.close()
    fileList = context[0 : -1].split('\n')

    cnt = 0
    cntFail = 0

    file = open('py150_new/' + datasetName + '.txt', 'w')
    for x in fileList:
        dir1 = x[5 : x.find('/', 5)]
        dir2 = x[x.find('/', 5) + 1 : x.find('/', x.find('/', 5) + 1)]
        assert os.path.isdir('py150_files/data/' + dir1 + '/' + dir2)
        assert os.path.isfile('py150_files/' + x)
        if (not os.path.isdir('py150_structures/' + dir1 + '/' + dir2)):
            continue
        assert(os.path.isfile('py150_structures/' + dir1 + '/' + dir2 + '/node.txt'))
        assert(os.path.isfile('py150_structures/' + dir1 + '/' + dir2 + '/edge.txt'))

        cnt += 1
        nodes, edges, rootId, rootChild = getGraph('py150_structures/' + dir1 + '/' + dir2 + '/', x)
        if (nodes == -1):
            cntFail += 1
            continue
        nodes, edges, rootChild = cutGraph(nodes, edges, rootId, rootChild)
        tokens, crossEdges = getCodeTokens('py150_files/' + x, rootChild)

        info = dict()
        info['path'] = x
        info['tokens'] = tokens
        info['nodes'] = nodes
        info['edges'] = edges
        info['crossEdges'] = crossEdges

        print(info, file = file)
        print('done', cnt - cntFail, 'using time ', round(time.time() - startTime, 2))

    file.close()
    print(cnt - cntFail, 'in total')


os.system('mkdir py150_new')
work('python50k_eval')
work('python100k_train')
