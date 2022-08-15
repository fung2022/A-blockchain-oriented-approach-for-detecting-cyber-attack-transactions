# -*- coding: utf-8 -*-

import igraph as ig
import numpy as np
import weighted_extended_isolation_forest as iso
import copy
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()

def branch2num(branch, init_root=0):
    num = [init_root]
    for b in branch:
        if b == 'L':
            num.append(num[-1] * 2 + 1)
        if b == 'R':
            num.append(num[-1] * 2 + 2)
    return num

def gen_graph(branches, g = None, init_root = 0, pre = ''):
    num_branches = [branch2num(i, init_root) for i in branches]
    all_nodes = [j for branch in num_branches for j in branch]
    all_nodes = np.unique(all_nodes)
    all_nodes = all_nodes.tolist()
    if g is None:
        g=ig.Graph()
    for k in all_nodes : g.add_vertex(pre+str(k))
    t=[]
    for j in range(len(branches)):
        branch = branch2num(branches[j], init_root)
        for i in range(len(branch)-1):
            pair = [branch[i],branch[i+1]]
            if pair not in t:
                t.append(pair)
                g.add_edge(pre+str(branch[i]),pre+str(branch[i+1]))
    return g,max(all_nodes)

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 500
x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
#Add manual outlier
x[0]=3.3
y[0]=3.3
X=np.array([x,y]).T


F = iso.iForest(X,ntrees=200, sample_size=256,ExtensionLevel=1)
S=F.compute_paths(X_in=X)
ss=np.argsort(S)

n_v = [0]
n_e = []
jt=0
T=F.Trees[jt]
gg,nn=gen_graph(iso.all_branches(T.root,[],None,),None,0,'0_')
n_v.append(gg.vcount())
n_e.append(gg.ecount())
vstyle={}
vstyle["vertex_size"]=[2.5]*gg.vcount()
vstyle["vertex_color"]=['black']*gg.vcount()
#vstyle["vertex_label"]=g.vs['name']
vstyle["vertex_label_dist"]=2
vstyle["bbox"] = (400, 400)
vstyle["edge_color"]= [(0,0.,0.)]*gg.ecount()
vstyle["edge_width"] = [0.4]*gg.ecount()
vstyle["layout"]=gg.layout_reingold_tilford(root=[0])
vstyle["edge_curved"]=0.00
vstyle["margin"]=10
ig.plot(gg,**vstyle)

"""indentification of outlier"""
depth = 50
for i in range(len(X)):
    P=iso.PathFactor(X[i],T)
    Gn=branch2num(P.path_list)
    if len(Gn)< depth:
        depth = len(Gn)
        outlier_index = i

"""indentification of normal observation"""
depth = 0
for i in range(len(X)):
    P=iso.PathFactor(X[i],T)
    Gn=branch2num(P.path_list)
    if len(Gn) >= depth:
        depth = len(Gn)
        normal_index = i

"""Single tree of an outlier and normal observation"""
P1=iso.PathFactor(X[outlier_index],T)
Gn1=branch2num(P1.path_list)
lb1=gg.get_shortest_paths('0_'+str(Gn1[0]), '0_'+str(Gn1[-1]))[0]
le1=gg.get_eids([(lb1[i],lb1[i+1]) for i in range(len(lb1)-1)])
vstyle2 = copy.deepcopy(vstyle)
for j in le1: 
    vstyle2["edge_color"][j]= 'red'
    vstyle2["edge_width"][j] = 1.9
for v in lb1:
    vstyle2["vertex_color"][v]='red'
ig.plot(gg,**vstyle2)

P2=iso.PathFactor(X[normal_index],T)
Gn2=branch2num(P2.path_list)
lb2=gg.get_shortest_paths('0_'+str(Gn1[0]), '0_'+str(Gn2[-1]))[0]
le2=gg.get_eids([(lb2[i],lb2[i+1]) for i in range(len(lb2)-1)])
vstyle3 = copy.deepcopy(vstyle2)
for x in le2: 
    vstyle3["edge_color"][x]= 'blue'
    vstyle3["edge_width"][x] = 1.9
for y in lb2:
    vstyle3["vertex_color"][y]='blue'
ig.plot(gg,**vstyle3)

"""comparision of tree depth within the weighted and extended isolation"""
for kt in range(1,100):
    T=F.Trees[kt]
    gg,nn=gen_graph(iso.all_branches(T.root,[],None,),gg,0,str(kt)+'_')
    n_v.append(gg.vcount())
    n_e.append(gg.ecount())
    
vstyle={}
vstyle["vertex_size"]=[.01]*gg.vcount()
vstyle["vertex_color"]=['black']*gg.vcount()
#vstyle["vertex_label"]=g.vs['name']
vstyle["vertex_label_dist"]=2
vstyle["bbox"] = (700, 700)
vstyle["edge_color"]= [(0,0.,0.)]*gg.ecount()
vstyle["edge_width"] = [0.05]*gg.ecount()
vstyle["layout"]=gg.layout_reingold_tilford_circular(root=n_v)
vstyle["edge_curved"]=0.00
vstyle["margin"]=10
ig.plot(gg,**vstyle)

vstyle2 = copy.deepcopy(vstyle)
for kt in range(0,100):
    T=F.Trees[kt]
    pre=str(kt)+'_'
    P=iso.PathFactor(X[outlier_index],T)
    Gn=branch2num(P.path_list)
    lb=gg.get_shortest_paths(pre+str(Gn[0]), pre+str(Gn[-1]))[0]
    le=gg.get_eids([(lb[i],lb[i+1]) for i in range(len(lb)-1)])
    for j in le: 
        vstyle2["edge_color"][j]= 'red'
        vstyle2["edge_width"][j] = 1.5
ig.plot(gg,**vstyle2)

vstyle3 = copy.deepcopy(vstyle2)
for kt in range(0,100):
    T=F.Trees[kt]
    pre=str(kt)+'_'
    P=iso.PathFactor(X[normal_index],T)
    Gn=branch2num(P.path_list)
    lb=gg.get_shortest_paths(pre+str(Gn[0]), pre+str(Gn[-1]))[0]
    le=gg.get_eids([(lb[i],lb[i+1]) for i in range(len(lb)-1)])
    for j in le: 
        vstyle3["edge_color"][j]= 'blue'
        vstyle3["edge_width"][j] = 1.5
ig.plot(gg,**vstyle3)



