import inspect
import argparse
import json
import logging
import os, sys
from multiprocessing import Process, Barrier
import uuid
import pickle as pk
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from phm_framework.utils import get_model_creator
from phm_framework.trainers.fen import FENTrainer
from phm_framework.trainers.rna_mas import RNAMASTrainer
from phm_framework.data.generators import Sequence


logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

def train(opt_code, config, trainer, init, callback):
    print(config)
    for key in config.keys():
        if '__' in key:
            sect, param = key.split('__')
            config[sect][param] = config[key]

    config = {k: v for k, v in config.items() if '__' not in k}
    print(config)

    return phm_framework.optimization.utils.parameter_opt_cv(opt_code, config, trainer=trainer,
                                                             debug=args.debug, init=init,
                                                             callback=callback)

def get_config(callback, init, net_name, run_name):

    space = {}

    model_creator = get_model_creator(net_name)
    model_parameters = inspect.signature(model_creator).parameters.keys()
    hp_ranges = hp.compute_ranges(task)
    space.update({f"model__{k}": (i, e) for k, (i, e) in hp_ranges.items() if k in model_parameters})

    space.update({
        'train__lr': (1e-6, 1e-3),
        'train__ts_len': (min(64, task['min_ts_len'] - 1), min(512, task['min_ts_len'] - 1)),

    })
    hp.remove_fixed_params_ds(space, args.dataset)
    hp.remove_fixed_params(space, fixed_params)
    # points_to_evaluate = model_config['points_to_evaluate']

    optimization_config = {
        'model': {
            'net': net_name,
            'output_dim': hp.get_output_dim(task),
            'output': hp.get_output(task),

        },

        'data': {
            'dataset_name': args.dataset,
            'dataset_target': args.task,
            'preprocess': 'std',

        },

        'train': {
            'epochs': 1 if args.debug else 100,
            'batch_size': 128,
            'timeout': 60 * 30,
            'verbose': True,
            'num_folds': 2 if args.debug else min(5, max_folds),
            'ncpus': args.ncpus,
            'random_seed': 666,
            'cv_early_stopping': True,
            'monitor': 'val_mse',
            'score_threshold': 180
        },

        'log': {
            'directory': os.path.join(args.output, run_name),
            'save_only_best': True,
        },
    }

    if args.dataset in hp.DATASET_FIXED_PARAMS:
        optimization_config = hp.update_dict(hp.DATASET_FIXED_PARAMS[args.dataset], optimization_config, task)

    for k, v in fixed_params.items():
        optimization_config[k].update(v)

    return optimization_config, space


def train_composite_part(opt_code, run_name, trainer, init, callback, part_func, net_name):

    def _train(config):
        return train(opt_code, config, trainer, init, callback)

    optimization_config, space = get_config(callback, init, net_name, run_name)

    run_name = f'{run_name}#{net_name}'

    def trial_str_creator(trial):
        trialname = run_name + "_" + args.dataset + "_" + args.task + "_" + trial.trial_id
        return trialname

    ray.init(num_cpus=args.ncpus, num_gpus=args.ngpus)

    from ray.tune.search.bayesopt import BayesOptSearch

    def trial_str_creator(trial):
        trialname = run_name + "_" + args.dataset + "_" + args.task + "_" + trial.trial_id
        return trialname

    optimization_config, space, pte = part_func(optimization_config, space)

    bayesopt = ray.tune.search.bayesopt.BayesOptSearch(
        space=space, mode="min", metric="score", points_to_evaluate=pte,
        random_search_steps=100)

    csvlogger = ray.tune.logger.CSVLoggerCallback()

    ray.tune.run(
        _train,
        name=run_name + "_" + args.dataset + "_" + args.task,
        config=optimization_config,
        resources_per_trial={'gpu': 1 if args.ngpus > 0 else 0, 'cpu': min(2, args.ncpus)},
        num_samples=100,
        search_alg=bayesopt,
        callbacks=[csvlogger],
        log_to_file=False,
        trial_name_creator=trial_str_creator,
        storage_path=os.path.join(args.output, f"{run_name}/results/opt"),
    )

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-fm", "--fen_model", help="FEN model", choices=['mscnn', 'rnn', 'mscnn1d'],  type=str, required=True)
    parser.add_argument("-rm", "--rna_model", help="RNA MAS model", choices=['mscnn', 'rnn', 'mscnn1d'], type=str, required=True)
    parser.add_argument("-d", "--dataset", help="Dataset params", type=str, required=True)
    parser.add_argument("-t", "--task", help="Dataset task", type=str, required=True)
    parser.add_argument("-c", "--cuda", help="Cuda visible", choices=["0", "1"], default="", required=False)
    parser.add_argument("-nc", "--ncpus", help="CPUs to take", type=int, required=False, default=4)
    parser.add_argument("-ng", "--ngpus", help="GPUs to take", type=int, required=False, default=2)
    parser.add_argument("-b", "--debug", help="Debug mode", action='store_true')
    parser.add_argument("-o", "--output", help="Output dir", type=str, required=True)


    # Read arguments from command line
    args = parser.parse_args()

    logging.info("Params read")

    fixed_params = {}

    logging.info("GPU: " + str(args.cuda != ""))
    if args.cuda != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ncpus = args.ncpus
    logging.info(f"Limiting to tensorflow to use only {ncpus} threads")

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_MAX_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(ncpus)
    os.environ['RAY_memory_monitor_refresh_ms'] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    tf.config.threading.set_inter_op_parallelism_threads(
        ncpus
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        ncpus
    )
    tf.config.set_soft_device_placement(True)


    import phm_framework
    from phm_framework.optimization import hyper_parameters as hp
    import ray
    from phm_framework.data import datasets as phmd


    data_meta = phmd.read_meta(args.dataset)
    task = phmd.get_task(data_meta, args.task)
    max_folds = task['num_units'] - 1 if 'min_units_per_class' not in task else task['min_units_per_class'] - 1
    

    if max_folds <= 1: # split time series
        max_folds = 3

    
    print("Folds: ", max_folds)

    fen_op_code = "fen_" + uuid.uuid4().hex
    rnamas_op_code = "rnamas_" + uuid.uuid4().hex


    # callbacks to synchronize the two BO processes
    def fen_callback(results, output_dir):
        flag_dir = os.path.join(args.output, "flags")
        if not os.path.exists(flag_dir):
            os.makedirs(flag_dir)


        flag_file = os.path.join(flag_dir, f"fen_{fen_op_code}_finished.flag")
        fp = open(flag_file, 'w')
        fp.write(output_dir)
        fp.close()

        time.sleep(10)
        while os.path.exists(flag_file):
            time.sleep(10)

    def init_rnamas_callback(experiment_config, output_dir):
        flag_dir = os.path.join(args.output, "flags")
        flag_file = os.path.join(flag_dir, f"fen_{fen_op_code}_finished.flag")
        while not os.path.exists(flag_file):
            time.sleep(10)

        fp = open(flag_file, 'r')
        fen_output_dir = fp.read()
        fp.close()


        print("Found FEN finished flag")
        rfile = os.path.join(fen_output_dir, f"best_{fen_op_code}_results.pk")
        fen_results = pk.load(open(rfile, 'rb'))
        experiment_config['model']['fen_results'] = fen_results

        return experiment_config

    def end_rnamas_callback(results, output_dir):
        flag_dir = os.path.join(args.output, "flags")
        flag_file = os.path.join(flag_dir, f"fen_{fen_op_code}_finished.flag")
        os.remove(flag_file)

    def fen_config(config, space):
        net = config['model']['net']
        pte = None
        return config, space, pte

    def rna_mas_config(config, space):
        if space is not None:
            del space['model__nblocks']

            if 'model__f1' in space:
                del space['model__f1']
                del space['model__f2']
                del space['model__f3']
                del space['model__msblocks']
                space['train__ts_len'] = (10, 30)
                config['model']['msblocks'] = 0

            del space['model__kernel_size']

            print("Adding stride param")
            space['train__stride'] = (1, 15)
        else:
            config['train']['stride'] = 1.5

        config['model']['nblocks'] = 4
        config['model']['fold_input'] = False
        config['model']['max_conv_filters'] = 32
        config['model']['kernel_size'] = (3, 3)

        net = config['model']['net']
        pte = None

        return config, space, pte

    # two parallel BO processes
    fen_process = Process(target=train_composite_part,
                          args=(fen_op_code, 'fen', FENTrainer(), None, fen_callback,
                                fen_config, args.fen_model))
    rna_process = Process(target=train_composite_part,
                          args=(rnamas_op_code, 'rna_mas', RNAMASTrainer(),
                                init_rnamas_callback, end_rnamas_callback,
                                rna_mas_config, args.rna_model))
    fen_process.start()
    rna_process.start()

    fen_process.join()
    rna_process.join()


