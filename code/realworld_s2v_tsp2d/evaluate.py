# Set True only if debugging in vscode, using debug configuration "s2v_mvc main.py"
DEBUG_CPP = False 
if DEBUG_CPP:
    import ptvsd
    # port=3000
    # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
    # ptvsd.wait_for_attach()
    # 5678 is the default attach port in the VS Code debug configurations
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()
import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import sys
import time
from tqdm import tqdm
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
        
sys.path.append( '%s/tsp2d_lib' % os.path.dirname(os.path.realpath(__file__)) )
from tsp2d_lib import Tsp2dLib
    
def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = 10000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = line.split(' ')
                it = int(line[1].strip())
                r = float(line[-1].strip())
                if r < best_r:
                    best_r = r
                    best_it = it
    assert best_it >= 0
    print 'using iter=', best_it, 'with r=', best_r
    return '%s/%s_iter_%d.model' % (opt['save_dir'], opt['sample_name'], best_it)

def GetGraph(fname, need_norm):
    norm = 1.0
    with open(fname, 'r') as f_tsp:
        coors = {}
        in_sec = False
        n_nodes = -1
        for l in f_tsp:
            if 'DIMENSION' in l:
                n_nodes = int(l.strip().split(' ')[-1].strip())
            if in_sec:
                idx, x, y = [w.strip() for w in l.strip().split(' ') if len(w.strip())]
                idx = int(idx)
		if np.fabs(float(x)) > norm:
		    norm = np.fabs(float(x))
		if np.fabs(float(y)) > norm:
		    norm = np.fabs(float(y))
                coors[idx - 1] = [float(x), float(y)]
                assert len(coors) == idx
                if len(coors) == n_nodes:
                    break
            elif 'NODE_COORD_SECTION' in l:
                in_sec = True
    
    if need_norm:
        for i in coors:
            coors[i][0] /= norm
            coors[i][1] /= norm
    assert len(coors) == n_nodes
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    nx.set_node_attributes(g, name='pos', values=coors)
    return g

def dist(coors, i, j):
    dx = float(coors[i][0] - coors[j][0])
    dy = float(coors[i][1] - coors[j][1])

    dd = np.sqrt(dx * dx + dy * dy)
    return np.round(dd)

def get_val(sol, g):
    t = []
    for i in range(sol[0]):
        t.append(sol[i + 1])

    if len(t) != nx.number_of_nodes(g):
        print len(t), nx.number_of_nodes(g)
    assert len(t) == nx.number_of_nodes(g)
    val = 0.0
    coors = nx.get_node_attributes(g, 'pos')
    for i in range(nx.number_of_nodes(g)):
        if i == nx.number_of_nodes(g) - 1:
            next = 0
        else:
            next = i + 1
        val += dist(coors, t[i], t[next])
    return val

if __name__ == '__main__':
    api = Tsp2dLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):        
        opt[sys.argv[i][1:]] = sys.argv[i + 1]
    
    # model_file = find_model_file(opt)
    # assert model_file is not None
    # print 'loading', model_file
    # sys.stdout.flush()
    # api.LoadModel(model_file)

    # Test all instances in ../../data/tsplib/
    tsp_files = os.listdir('../../data/tsplib')
    approx_ratio = []
    time_ratio = []
    s2v_sol = []
    s2v_time = []
    concorde_sol = []
    concorde_time = []

    tsp_file = opt['tsp_file']
    # for tsp_file in tsp_files:
    fname = '%s/%s/%s' % (opt['data_root'], opt['folder'], tsp_file)
    # fname = '%s/%s/%s' % (opt['data_root'], opt['folder'], opt['sample_name'])

    # Concorde:
    # fname = get_dataset_path("berlin52")
    solver = TSPSolver.from_tspfile(fname)
    t3 = time.time()
    solution = solver.solve()
    t4 = time.time()
    # print 'Concorde sol val:' , solution.optimal_value
    concorde_sol.append(solution.optimal_value)
    concorde_time.append(t4-t3)
    
    model_file = find_model_file(opt)
    assert model_file is not None
    print 'loading', model_file
    sys.stdout.flush()
    api.LoadModel(model_file)

    g_norm = GetGraph(fname, True)
    api.InsertGraph(g_norm, is_test=True)
    g_raw = GetGraph(fname, need_norm=False)
    
    test_name = tsp_file # opt['sample_name']
    result_file = '%s/test-%s-gnn-%s-%s.csv' % (opt['save_dir'], test_name, opt['min_n'], opt['max_n'])

    with open(result_file, 'w') as f_out:
        print 'testing'
        sys.stdout.flush()

        t1 = time.time()
        _, sol = api.GetSol(0, nx.number_of_nodes(g_norm))
        t2 = time.time()
        val = get_val(sol, g_raw)
        f_out.write('%d,' % val)
        f_out.write('%d' % sol[0])
        for i in range(sol[0]):
            f_out.write(' %d' % sol[i + 1])
        f_out.write(',%.6f\n' % (t2 - t1))

    # print 'average tour length: ', val
    s2v_sol.append(val)
    s2v_time.append(t2-t1)

    approx_ratio.append(s2v_sol[-1]/concorde_sol[-1])
    time_ratio.append(s2v_time[-1]/concorde_time[-1])
    
    
    print 'average approximation ratio:', np.mean(approx_ratio)
    print 'average solution time ratio:', np.mean(time_ratio)

    with open("tsplib_aprx_ratio.txt", "a") as myfile:
        myfile.write(str(val/solution.optimal_value)+"\n")
    with open("tsplib_time_ratio.txt", "a") as myfile:
        myfile.write(str(s2v_time[-1]/concorde_time[-1])+"\n")
    with open("tsplib_performance.txt", "a") as myfile:
        myfile.write(tsp_file + " " + str(nx.number_of_nodes(g_norm)) + " " +str(val/solution.optimal_value) +" "+ str(s2v_time[-1]/concorde_time[-1])+"\n")
    