import argparse
import inspect
import json
import logging
import os
import pickle
import shutil
import sys
import time

import numpy as np

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)


class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung

    def to_dict(self):
        out = {'parent': self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out


class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n, self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n], 'objective_val')]
        objective_vals = sorted(objective_vals, key=lambda x: x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)

    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir, 'results_tmp.pkl'), 'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save(epoch=self.model.epochs)

    def run(self):
        epochs = 0
        # self.get_eval_arch(1)
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            # If epoch has changed then evaluate the network.
            if epochs < self.model.epochs:
                epochs = self.model.epochs
                #self.get_eval_arch(1)
            if self.iters % 500 == 0:
                self.save()
        self.save()

    def fine_tune(self, arch, steps=0):
        
        if steps < 0:
            try:
                # evaluate using shared weights
                ppl = self.model.evaluate(arch)
            except Exception as e:
                ppl = 1000000
            return ppl
        
        elif steps > 0:
            self.model.load(epoch=49)
                
        for _ in range(steps):
            # tune the shared weights
            self.model.train_batch(arch)
            self.iters += 1
        
        try:
            ppl = self.model.evaluate(arch, split='valid')
        except Exception as e:
            ppl = 1000000     
        
        return ppl
            
    def local_search(self, num_init=10, steps=0, cycles=300):
        """
        Local search
        """
        print('starting local search')
        history = []  # Not used by the algorithm, only used to report results.
        initial_models = []

        # Initialize the search with random models.
        while len(history) < min(num_init, cycles):
            arch = self.model.sample_arch()
            valid_error = self.fine_tune(arch, steps=steps)
            history.append([arch, valid_error])
            initial_models.append([arch, valid_error])
            print('initial model:', valid_error)

        # initialize the first iteration using the best of the initial arches
        current_model = min(initial_models, key=lambda i: i[1])
        neighbors = self.model.get_nbhd(current_model[0])
        print('len nbrs:', len(neighbors))

        # TODO: this can be cleaned up
        while len(history) <= cycles:
            while True:
                neighbor_arch = neighbors.pop()
                neighbor_valid_error = self.fine_tune(neighbor_arch, steps=steps)
                print('new neighbor. nbrs left:', len(neighbors), 'error', neighbor_valid_error)
                history.append([neighbor_arch, neighbor_valid_error])

                if neighbor_valid_error < current_model[1]:
                    print('found better arch:', neighbor_valid_error, 'vs', current_model[1])
                    current_model = [neighbor_arch, neighbor_valid_error]
                    neighbors = self.model.get_nbhd(current_model[0])
                    print('len nbrs:', len(neighbors))
                    
                    with open(os.path.join(self.save_dir,
                                           'local_search_{}.obj'.format(len(history))), 'wb') as f:
                        pickle.dump(history, f)

                if len(history) >= cycles or len(neighbors) == 0:
                    break

            if len(history) >= cycles:
                break
            
            arch = self.model.sample_arch()
            valid_error = self.fine_tune(arch, steps=steps)
            current_model = [arch, valid_error]
            print('reached local min. new arch:', current_model[1])
            history.append(current_model)
            neighbors = self.model.get_nbhd(current_model[0])
            print('len nbrs:', len(neighbors))

        print('start full history')
        print(history)

        with open(os.path.join(self.save_dir,
                               'local_search_{}.obj'.format(len(history))), 'wb') as f:
            pickle.dump(history, f)
                
        return history

        
    def get_eval_arch(self, rounds=None, inner_rounds=5):
        # n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1, int(self.B / 10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            print('starting round', r)
            sample_vals = []
            for k in range(inner_rounds):
                print('round', r, 'inner', k)
                
                # sample a random architecture
                arch = self.model.sample_arch()
                try:
                    # evaluate it using shared weights (really quick?)
                    ppl = self.model.evaluate(arch)
                except Exception as e:
                    ppl = 1000000
                # logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))

            # Save sample validations
            with open(os.path.join(self.save_dir,
                                   'sample_val_architecture_epoch_{}.obj'.format(self.model.epochs)), 'wb') as f:
                pickle.dump(sample_vals, f)

            sample_vals = sorted(sample_vals, key=lambda x: x[1])

            full_vals = []
            # train the top five on the full validation set
            if 'split' in inspect.getfullargspec(self.model.evaluate).args:
                for i in range(5):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x: x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (
                    ' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])

            # Save the fully evaluated architectures
            with open(os.path.join(self.save_dir,
                                   'full_val_architecture_epoch_{}.obj'.format(self.model.epochs)), 'wb') as f:
                pickle.dump(full_vals, f)
        return best_rounds


def main(args):
    # Fill in with root output path
    root_dir = os.getcwd()
    print('root_dir', root_dir)
    if args.save_dir is None:
        save_dir = os.path.join(root_dir, 'experiments/random_ws/ss_{}_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"),
                                                                                     args.search_space, args.seed))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.eval_only:
        assert args.save_dir is not None

    # Dump the config of the run folder
    with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
        json.dump(args.__dict__, fp)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    if args.benchmark == 'ptb':
        raise ValueError('PTB not supported.')
    else:
        data_size = 25000
        time_steps = 1
    
    if args.benchmark == 'cnn':
        from optimizers.random_search_with_weight_sharing.darts_wrapper_discrete import DartsWrapper
        model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs,
                             num_intermediate_nodes=search_space.num_intermediate_nodes, search_space=search_space,
                             init_channels=args.init_channels, cutout=args.cutout)
    else:
        raise ValueError('Benchmarks other cnn on cifar are not available')
    
    B = int(args.epochs * data_size / args.batch_size / time_steps)
    # B = epochs * 390
    # B = 3    

    searcher = Random_NAS(B, model, args.seed, save_dir)
    logging.info('budget: %d' % (searcher.B))

    if True:
        # load the supernet
        searcher.model.load(epoch=49)
        
        # run local search
        ls_epochs = 0
        steps = int(ls_epochs * data_size / args.batch_size)
        archs = searcher.local_search(num_init=10, steps=steps, cycles=300)
        
    else:
        
        # train supernet
        if not args.eval_only:
            logging.info('starting searcher.run')
            searcher.run()
            logging.info('finished searcher.run')
            archs = [[0]]
            #archs = searcher.get_eval_arch(rounds=2)
            #logging.info('finished get_eval_arch')
            print('num archs:', len(archs))
        else:
            np.random.seed(args.seed + 1)
            archs = searcher.get_eval_arch(2)
        logging.info('printing archs')    
        logging.info(archs)
        arch = None
        #logging.info('saving archs')    
        #arch = ' '.join([str(a) for a in archs[0][0]])
        #with open('/tmp/arch', 'w') as f:
        #    f.write(arch)
        return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    # /home/ubuntu/nasbench-1shot1/experiments/ft0_nov29/
    # '/home/ubuntu/nasbench-1shot1_crwhite/experiments/random_ws/ss_20201130-033347_1_0'
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    # CIFAR-10 only argument.  Use either 16 or 24 for the settings for random_ws search
    # with weight-sharing used in our experiments.
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    args = parser.parse_args()

    main(args)
