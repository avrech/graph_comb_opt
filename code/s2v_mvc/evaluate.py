# Set True only if debugging in vscode, using debug configuration "s2v_mvc main.py"
VSCODE_DEBUG_MODE = True 
if VSCODE_DEBUG_MODE:
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

sys.path.append( '%s/mvc_lib' % os.path.dirname(os.path.realpath(__file__)) )
from mvc_lib import MvcLib
import sys
sys.path.append("/home/daniela/PycharmProjects/cplex-samples")
from mipex1_mvc import CplexMvcSolver
# from mipex1_mvc import CplexMvcSolver

def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = 1000000
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
    return '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], min_n, max_n, best_it)
    
if __name__ == '__main__':
    api = MvcLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    model_file = find_model_file(opt)
    assert model_file is not None
    print 'loading', model_file
    sys.stdout.flush()
    api.LoadModel(model_file)

    n_test = 1000
    f = open(opt['data_test'], 'rb')
    frac = 0.0

    test_name = opt['data_test'].split('/')[-1]
    result_file = '%s/test-%s-gnn-%s-%s.csv' % (opt['save_dir'], test_name, opt['min_n'], opt['max_n'])
    ref_cplex_val = []
    ref_cplex_sol = []
    ref_cplex_val_sum = 0
    with open(result_file, 'w') as f_out:
        print 'testing'
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = cp.load(f)
            api.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = api.GetSol(i, nx.number_of_nodes(g))
            t2 = time.time()
            f_out.write('%.8f,' % val)
            f_out.write('%d' % sol[0])
            for i in range(sol[0]):
                f_out.write(' %d' % sol[i + 1])
            f_out.write(',%.6f\n' % (t2 - t1))
            frac += val
            # Compute cplex solution:
            cplex_mvc_solver = CplexMvcSolver(g)
            cplex_val, cplex_sol = cplex_mvc_solver.solve()
            ref_cplex_sol.append(cplex_sol)
            ref_cplex_val.append(cplex_val)
            ref_cplex_val_sum += cplex_val
    print 'average size of vc: ', frac / n_test
    print 'average size of cplex vc: ', ref_cplex_val_sum / n_test
    print 'approximation ratio s2v/cplex: ', frac / ref_cplex_val_sum
