import os
from typing import Callable
import numpy as np
from ray import train as rtrain
import multiprocessing
import sys
import logging
import traceback
import pickle as pk
import phm_framework
from phm_framework.data import datasets as phmd
from phm_framework.logging import secure_decode, get_best_info
from phm_framework.trainers.utils import get_task

logging.basicConfig(level=logging.INFO)


def parameter_opt_cv(opt_code: str,
                     experiment_config: dict = {},
                     trainer=None,
                     debug: bool=False,
                     init = None,
                     callback = None):
    try:

        training_config = experiment_config['train']
        


        output_dir = experiment_config['log']['directory']

        net_name = experiment_config['model']['net']
        data_name = experiment_config['data']['dataset_name']
        target = experiment_config['data']['dataset_target']

        path_net_name = net_name
        if isinstance(net_name, list):
            path_net_name = '__'.join(net_name)

        output_dir = os.path.join(output_dir, data_name, target, path_net_name)

        print("Training config", training_config)
        if init is not None:
            print("Found init callback")
            experiment_config = init(experiment_config, output_dir)


        data_meta = phmd.read_meta(data_name)

        def get_model_creator(net_name):

            net_creator_func = f"create_model"
            if isinstance(net_name, list):
                net_name = net_name[0]
            net_module = getattr(getattr(phm_framework, 'nets'), net_name)
            creator = getattr(net_module, net_creator_func)

            return creator

        task = get_task(data_meta, target, get_model_creator(net_name))

        # min_score = config.pop('min_score')
        monitor = secure_decode(training_config, "monitor", str, default='val_loss', task=task, pop=False)
        timeout = secure_decode(training_config, "timeout", int, default=None, task=task)
        num_folds = secure_decode(training_config, 'num_folds', int, default=5, task=task, pop=False)
        cv_early_stopping = secure_decode(training_config, 'cv_early_stopping', bool, default=True,
                                          task=task, pop=False)
        score_threshold = secure_decode(training_config, 'score_threshold', int, default=5, task=task, pop=False)

        experiment_config['train'] = training_config


        data = experiment_config.copy()
        data['model'] = path_net_name #model_creator.__name__
        data['folds'] = {}

        # cross-validation
        finish = False
        results = []

        logging.info(f"Opt :: Train with {num_folds} folds")
        for ifold in range(num_folds):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=trainer.train, args=(experiment_config, ifold,
                                                              queue, debug, output_dir, timeout))

            p.start()
            p.join()
            logging.info(f"Opt :: Finished fold {ifold}")
            r = queue.get()
            results.append(r)

            if r is None:
                finish = True

            else:
                data['folds'][ifold] = r[0]

            if len(data['folds'].keys()) > 0:
                # compute the mean score
                epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys()]
                scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys()]

                rtrain.report({"score": np.mean(scores), "mean_epochs": np.mean(epochs),
                               "std_score": np.std(scores)})

            elif finish:
                logging.info("Opt :: Not finished any trial")
                rtrain.report({"score": 999, "mean_epochs": 999, "std_score": 999, "nasa_score": 999, "mae": 999})

            if finish:
                logging.info("Opt :: Finished train")
                return




            # compute the mean score
            epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys()]
            scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys()]

            best_arch_hash, best_score, best_std = get_best_info(experiment_config['model']['net'],
                                                                 data['data']['dataset_name'],
                                                                 monitor,
                                                                 output_dir)

            if cv_early_stopping:

                if np.mean(scores) > score_threshold:
                    logging.info(
                        f"Opt :: Stopping because current mean score {np.mean(scores)} "
                        f"is higher than the score threshold of {score_threshold}")
                    break

                alpha = 0.5
                # stopping criteria
                if alpha * np.mean(scores) >= best_score:
                    logging.info(
                        f"Opt :: Stopping because the {alpha * 100}% of the current mean score {np.mean(scores)} "
                        f"is higher than best score {best_score} found")
                    break

                if len(scores) > 1 and (alpha * np.std(scores) >= best_std):
                    logging.info(
                        f"Opt :: Stopping because the {alpha * 100}% of the current std score {np.mean(scores)} "
                        f"is higher than the std {best_std} of the best score found")
                    break

        rfile = os.path.join(output_dir, f'best_{opt_code}_results.pk')
        best_results = None
        if os.path.exists(rfile):
            best_results = pk.load(open(rfile, 'rb'))

            if best_results[0][1] != best_arch_hash:
                print("New Best Results:", results)
                pk.dump(results, open(rfile, 'wb'))
                best_results = results
        elif len(results) == num_folds:
            print("New Best Results:", results)
            pk.dump(results, open(rfile, 'wb'))
            best_results = results


        # keep only best model
        if experiment_config['log']['save_only_best']:

            # remove worst models
            for file in filter(lambda f: (best_arch_hash not in f) and '.h5' in f, os.listdir(output_dir)):
                os.remove(os.path.join(output_dir, file))

        rtrain.report({"score": np.mean(scores), "mean_epochs": np.mean(epochs), "std_score": np.std(scores)})

        if callback is not None and best_results is not None:
            callback(best_results, output_dir)

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

