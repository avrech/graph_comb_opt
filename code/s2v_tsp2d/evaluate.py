import pickle
import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../pyconcorde'))
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
sys.path.append( '%s/tsp2d_lib' % os.path.dirname(os.path.realpath(__file__)) )
from tsp2d_lib import Tsp2dLib
    
TESTSET = 'nr_test_tsp200_norm1e6' # 'tsplib' | 'tsp2d' | 'nr_test_tsp_norm1e6' | 'nr_test_tsp200_norm1e6'

def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = 10000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = filter(None, line.split(' '))
                it = int(line[1].strip())
                r = float(line[line.index('length:')+1].strip())
                if r < best_r:
                    best_r = r
                    best_it = it
    assert best_it >= 0
    print('using iter=', best_it, 'with r=', best_r)
    return '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], min_n, max_n, best_it)

def TestSet():
    if TESTSET == 'tsp2d':
        folder = '%s/test_tsp2d/tsp_min-n=%s_max-n=%s_num-graph=1000_type=%s' % (opt['data_root'], opt['test_min_n'], opt['test_max_n'], opt['g_type'])
        with open('%s/paths.txt' % folder, 'r') as f:
            for line in f:
                fname = '%s/%s' % (folder, line.split('/')[-1].strip())
                coors = {}
                in_sec = False
                n_nodes = -1
                with open(fname, 'r') as f_tsp:
                    for l in f_tsp:
                        if 'DIMENSION' in l:
                            n_nodes = int(l.split(' ')[-1].strip())
                        if in_sec:
                            idx, x, y = [int(w.strip()) for w in l.split(' ')]
                            coors[idx - 1] = [float(x) / 1000000.0, float(y) / 1000000.0]
                            assert len(coors) == idx
                        elif 'NODE_COORD_SECTION' in l:
                            in_sec = True
                assert len(coors) == n_nodes
                g = nx.Graph()
                g.add_nodes_from(range(n_nodes))
                nx.set_node_attributes(g, name='pos', values=coors)
                yield g, fname            

    else: # TESTSET == 'tsplib' | 'nr_test_tsp_norm1e6' | 'nr_test_tsp200_norm1e6':
        folder = os.path.join('/home/daniela/avrech/technion/049053/graph_comb_opt/data', TESTSET)
        files =  os.listdir(folder)
        for fname in files:
            coors = {}
            in_sec = False
            n_nodes = -1
            tsp_file = os.path.join(folder,fname)
            node_idx = 0
            with open(tsp_file, 'r') as f_tsp:
                for l in f_tsp:
                    if 'EOF' not in l:
                        if 'DIMENSION' in l:
                            n_nodes = int(l.split(' ')[-1].strip())
                        if in_sec and len(l.strip().split(' '))>=3:
                            idx, x, y = [w.strip() for w in l.strip().split(' ') if w != '']
                            idx = int(idx)
                            # idx, x, y = [w.strip() for w in l.split(' ')]
                            coors[idx - 1] = [float(x) / 1000000.0, float(y) / 1000000.0]
                            assert len(coors) == idx
                            # if node_idx == n_nodes-1:
                            #     break
                            # node_idx += 1
                        elif 'NODE_COORD_SECTION' in l:
                            in_sec = True
            assert len(coors) == n_nodes
            g = nx.Graph()
            g.add_nodes_from(range(n_nodes))
            nx.set_node_attributes(g, name='pos', values=coors)
            yield g, tsp_file            

    
if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    if int(opt['test_all']) == 0:
        print('Testing only ', opt['save_dir'])
        model_dirs = [opt['save_dir']]
    else:
        print('Testing all trained models in results/')
        arch = os.path.basename(opt['save_dir'])
        results_content_paths = [os.path.join('results', elem) for elem in os.listdir('results')]
        model_dirs = [os.path.join(model_dir, arch) for model_dir in results_content_paths if os.path.isdir(model_dir)]

    '''  
    For each model in model_dirs, perform the loop.
    Concorde is evaluated only on the first loop.
    The results are stored in a dictionary:
    all_results[model-name][instance-name]['len' | 'time']
    '''
    res_fname = TESTSET + '-all-results.pkl'
    if os.path.isfile(res_fname):
        with open(res_fname, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = {}
        all_results['concorde'] = {}

    eval_concorde = True

    api = Tsp2dLib(sys.argv)

    for model_dir in model_dirs:
        # Load the model:
        opt['save_dir'] = model_dir
        opt['min_n'] = model_dir.split('-')[2]
        opt['max_n'] = model_dir.split('-')[3]
        model_name = opt['save_dir'].split('/')[1]
        if model_name not in all_results.keys():
            all_results[model_name] = {}

        model_file = find_model_file(opt)
        assert model_file is not None
        print('loading', model_file)
        sys.stdout.flush()
        api.LoadModel(model_file)

        # internal
        test_name = '-'.join([opt['g_type'], opt['test_min_n'], opt['test_max_n']])
        result_file = '%s/test-%s-gnn-%s-%s.csv' % (opt['save_dir'], test_name, opt['min_n'], opt['max_n'])

        # approx_ratio = []
        # time_ratio = []
        # s2v_sol = []
        # s2v_time = []
        # concorde_sol = []
        # concorde_time = []

        n_test = 1000
        frac = 0.0
        # s2v_concorde_results = {} # results per city in testset.

        # if os.path.exists(TESTSET + "_" + opt['g_type'] + "_performance.txt"):
        #     os.remove(TESTSET + "_" + opt['g_type'] + "_performance.txt")
        # if os.path.exists(TESTSET + "_" + opt['g_type'] + "_time_ratio.txt"):
        #     os.remove(TESTSET + "_" + opt['g_type'] + "_time_ratio.txt")
        # if os.path.exists(TESTSET + "_" + opt['g_type'] + "_aprx_ratio.txt"):
        #     os.remove(TESTSET + "_" + opt['g_type'] + "_aprx_ratio.txt")
        with open(result_file, 'w') as f_out:
            print('testing')
            sys.stdout.flush()
            idx = 0
            inst_idx = 1
            for g, fname in tqdm(TestSet(), 'Solving instance no. {}'.format(inst_idx)):
                inst_idx += 1
                testset_name = TESTSET if TESTSET != 'tsp2d' else "tsp2d_" + opt['g_type']
                inst_name = fname if TESTSET != 'tsp2d' else "inst_" + str(idx)
                inst_key = os.path.basename(inst_name)[:-4]
                if eval_concorde and inst_key not in all_results['concorde'].keys():
                    # Concorde:
                    # fname = get_dataset_path("berlin52")
                    solver = TSPSolver.from_tspfile(fname)
                    t3 = time.time()
                    solution = solver.solve(verbose=False)
                    t4 = time.time()
                    # print 'Concorde sol val:' , solution.optimal_value
                    # 1000000.0 is a normalization factor of s2v in TestSet above.
                    # concorde_sol.append(solution.optimal_value / 1000000.0)
                    # concorde_time.append(t4-t3)
                    all_results['concorde'][inst_key] = {
                        'len': solution.optimal_value / 1000000.0,
                        'time': t4-t3
                    }
                if inst_key not in all_results[model_name].keys():
                    api.InsertGraph(g, is_test=True)
                    t1 = time.time()
                    val, sol = api.GetSol(idx, nx.number_of_nodes(g))
                    t2 = time.time()

                    # s2v_sol.append(val)
                    # s2v_time.append(t2-t1)
                    # approx_ratio.append(s2v_sol[-1]/concorde_sol[-1])
                    # time_ratio.append(s2v_time[-1]/concorde_time[-1])

                    # f_out.write('%.8f,' % val)
                    # f_out.write('%d' % sol[0])
                    # for i in range(sol[0]):
                    #     f_out.write(' %d' % sol[i + 1])
                    # f_out.write(',%.6f\n' % (t2 - t1))
                    frac += val


                    # with open(testset_name + "_aprx_ratio.txt", "a") as myfile:
                    #     myfile.write(str(val/concorde_sol[-1])+"\n")
                    # with open(testset_name + "_time_ratio.txt", "a") as myfile:
                    #     myfile.write(str(s2v_time[-1]/concorde_time[-1])+"\n")
                    # with open(testset_name + "_performance.txt", "a") as myfile:
                    #     myfile.write(inst_name + " " + str(nx.number_of_nodes(g)) + " " +str(val/concorde_sol[-1]) +" "+ str(s2v_time[-1]/concorde_time[-1])+"\n")
                    # with open("results/" + testset_name + "_s2v_and_concorde_results.txt", "a") as myfile:
                    #     myfile.write(inst_name + " s2v_value: " + str(val) + " s2v_time: " + str(s2v_time[-1])
                    #                 + " concorde_value: " + str(concorde_sol[-1])
                    #                 + " concorde_time: " + str(concorde_time[-1]) + "\n")
                    idx += 1
                    # s2v_concorde_results[inst_key] = {
                    #     's2v_len': val,
                    #     's2v_time': s2v_time,
                    #     'concorde_len': concorde_sol[-1],
                    #     'concorde_time': concorde_time[-1]
                    # }

                    all_results[model_name][inst_key] = {
                        'len': val,
                        'time': t2-t1
                    }

        # with open(TESTSET + '_concorde_s2v_40_50_results.pkl', 'wb') as f:
        #     pickle.dump(s2v_concorde_results, f)

        print('average tour length: ', frac / n_test)
        eval_concorde = False
        # Save at the end of each iteration, to reduce damage in crash:
        with open(res_fname, 'wb') as f:
            pickle.dump(all_results, f)
        print('Save results dictionary to:', res_fname)
