import copy
import importlib
import logging
import sys
import time
import traceback
import pandas as pd

import phm_framework
import numpy as np
from phm_framework.data import datasets as phmd
from phm_framework.data.generators import Sequence
from phm_framework.logging import HASH_EXCLUDE, confighash, secure_decode, log_train
from phm_framework.nets.metrics import TimeStopping, AdditionalRULValidationSets
from phm_framework.optimization.hyper_parameters import RANGES
from phm_framework.utils import flat_dict, get_model_creator
from phm_framework.trainers.utils import get_task
import pickle as pk
from itertools import product

logging.basicConfig(level=logging.INFO)

SNRS = [-4, -2, 0, 2, 4, 6, 8, 10, None]




class FENTrainer:

    def train(self, config, ifold, queue, debug, directory, timeout, data_generators=None, compute_test=True):



        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from phm_framework.data import datasets as phmd
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        from phm_framework import nets
        from phm_framework.optimization import hyper_parameters as hp

        base_config = copy.deepcopy(config)

        try:

            # prepare output directory

            results_col = []

            logging.info('Trainer :: Starting training (fold %d) %s' % (ifold, config))

            config = copy.deepcopy(base_config)

            context = {}
            context['config'] = config
            context['config']['train']['fold'] = ifold

            training_config = config['train']
            net_config = config['model']
            data_config = config['data']

            net_name = net_config['net']
            path_net_name = net_name
            data_name = data_config['dataset_name']
            data_target = data_config['dataset_target']

            random_seed = training_config['random_seed']

            data_meta = phmd.read_meta(data_name)
            model_creator = get_model_creator(net_name)
            task = get_task(data_meta, data_target, model_creator)
            context['task'] = task
            context['debug'] = debug

            num_folds = config['train']['num_folds']

            csv_config = flat_dict(config.copy())
            if 'train__epochs' in csv_config:
                csv_config['train__max_epochs'] = csv_config.pop('train__epochs')
            csv_config['train__fold'] = ifold
            nhash = confighash(csv_config, exclude=HASH_EXCLUDE )
            arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold", "train__base_iterations",
                                                                       "train__num_folds", "train__internal_folds",
                                                                       "model__fen_results", "train__max_epochs",
                                                                       "train__verbose", "train__random_seed"])
            csv_config['run_hash'] = nhash
            csv_config['arch_hash'] = arch_hash

            tf.random.set_seed(random_seed)

            # prepare output directory
            if not os.path.exists(directory):
                os.makedirs(directory)

            net_path = f"{directory}/fen_net_{path_net_name}_{data_name}_{task['target']}_{arch_hash}_{nhash}.h5"
            net_history = f"{directory}/fen_net_{path_net_name}_{data_name}_{task['target']}_{arch_hash}_{nhash}_history.pk"


            # data reading and prepare data generators
            logging.info("Trainer :: Reading data")
            ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
            print("TS_LEN:", ts_len)

            # if already train, return saved history
            print(net_path)
            if os.path.exists(net_history) and os.path.exists(net_path):
                history = pk.load(open(net_history, 'rb'))
                if queue is not None:
                    queue.put((history, arch_hash, {'net_path': net_path,
                                                    'ts_len': ts_len,
                                                    'net_hash': nhash,
                                                    }))
                return

            """
            csv_config['val_mse'] = np.random.uniform()
            log_train(csv_config, directory)

            queue.put(({'val_mse': [np.random.uniform()]},
                       arch_hash, {'net_path': net_path,
                                            'ts_len': ts_len,
                                            'net_hash': nhash,
                                            }))

            return
            """

            base_iterations = secure_decode(training_config, "base_iterations", dtype=int, task=task, default=1)
            context['config']['train']['fen_ts_len'] = ts_len

            preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()
            context['config']['data']['preprocess'] = preprocess

            batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
            context['config']['train']['batch_size'] = batch_size

            datagen = secure_decode(data_config, "data_generator", dtype=str, task=task, default=Sequence)
            context['config']['data']['data_gen'] = datagen

            val_test_datagen = datagen
            context['config']['data']['val_test_data_gen'] = val_test_datagen

            extra_channel = getattr(importlib.import_module(model_creator.__module__), 'EXTRA_CHANNEL')
            context['config']['data']['extra_channel'] = extra_channel


            context['config']['train']['samples_per_unit'] = None

            context['config']['data']['generator_params'] = {
                'ts_len': ts_len,
                'extra_channel': extra_channel,
                'batch_size': batch_size,
            }

            # prepare data generators
            if data_generators is None:
                data_generators = phmd.load_generators(context, return_test=True)
            context['data_generators'] = data_generators

            logging.info("Trainer :: Finished Data reading")

            train_gen = data_generators['train']
            val_gen = data_generators['val']
            others_gen = [g for n, g in data_generators.items() if n != 'train']

            train_gen.batch_size = batch_size
            train_gen.extra_channel = extra_channel
            for gen in others_gen:
                gen.batch_size = 256
                gen.extra_channel = extra_channel

            # get input shape
            input_shape = self.get_input_shape(train_gen)

            # training config
            epochs = secure_decode(training_config, "epochs", int, task=task)
            epochs = min(5, epochs) if debug else epochs

            lr = secure_decode(training_config, "lr", float, pop=False, task=task)
            monitor = secure_decode(training_config, "monitor", str, default="val_loss", task=task)
            verbose = secure_decode(training_config, "verbose", bool, default=False, task=task)

            output_dim = hp.get_output_dim(task)
            output = hp.get_output(task)

            # create and compile model
            model_params = nets.get_model_params(net_config, model_creator, task)
            csv_config.update(flat_dict({'model': model_params}))
            csv_config['model__output'] = output
            csv_config['model__output_dim'] = output_dim
            csv_config['model__input_shape'] = input_shape
            del model_params['output_dim']
            del model_params['input_shape']
            del model_params['output']


            model = model_creator(input_shape, output_dim=output_dim, output=output,
                                  **model_params)

            context['model'] = model

            extra_callbacks = []
            if 'extra_callbacks' in training_config and training_config['extra_callbacks'] is not None:
                extra_callbacks = training_config['extra_callbacks'](context)

            results, history = self.__train(epochs, extra_callbacks, lr, model, monitor, task, timeout,
                                            train_gen, val_gen, verbose, context, compute_test=compute_test)

            results_col.append(results)

            csv_config.update({f"{k}": v for k, v in results.items()})
            csv_config["train__status"] = "BASE_FINISHED"

            log_train(csv_config, directory)


            base_results = pd.DataFrame(results_col).mean().to_dict()

            # save csv results
            #csv_config.update({f"base__{k}": v for k, v in results.items()})

            # save training history
            pk.dump(history, open(net_history, 'wb'))

            # save model
            model.save(net_path)


            if queue is not None:
                queue.put((history, arch_hash, {'net_path': net_path,
                                                'ts_len': ts_len,
                                                'net_hash': nhash,
                                               }))


            logging.info("Trainer :: Finished train")

            return base_results, arch_hash

        except Exception as ex:
            if 'OOM' in str(ex):
                csv_config["train__status"] = "OOM ERROR"
            else:
                csv_config["train__status"] = "ERROR: " + str(ex)

            logging.error("Error: %s" % ex)
            logging.error(traceback.format_exc())
            sys.stdout.flush()
            if queue is not None:
                queue.put(None)

            log_train(csv_config, directory)

    def __train(self, epochs, extra_callbacks, lr, model, monitor, task, timeout, train_gen, val_gen, verbose,
                context, compute_test=True):
        import tensorflow as tf
        from phm_framework.optimization import hyper_parameters as hp

        results = {}

        val_gen10 = val_gen.clone()
        val_gen10.ts_consider = 0.1

        val_gen20 = val_gen.clone()
        val_gen20.ts_consider = 0.2

        val_gen30 = val_gen.clone()
        val_gen30.ts_consider = 0.3

        val_gen0 = val_gen.clone()
        val_gen0.ts_consider = 0

        extra_callbacks.append(AdditionalRULValidationSets([(val_gen0, 'val_last'),  (val_gen10, 'val10'), (val_gen20, 'val20'), (val_gen30, 'val30')]))

        loss_metrics = hp.get_loss(task)
        model.compile(loss=loss_metrics[0],
                      metrics=loss_metrics[1:],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=False)
        logging.info("Trainer :: Model created")
        model.summary(print_fn=lambda x, *args, **kwargs: logging.info(x))
        # train
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ts = TimeStopping(seconds=timeout, verbose=1)
        logging.info("Trainer :: Started training")
        start_time = time.time()
        history = model.fit(train_gen, validation_data=val_gen,
                            epochs=epochs, verbose=(2 if verbose else 0),
                            callbacks=extra_callbacks + [es, ts, rlr])
        history = history.history

        # save csv results
        results['train__time'] = (time.time() - start_time)
        results.update({k: history[k][-1] for k in history.keys() if k.startswith('val')})

        if compute_test:
            context['model'] = model
            test_metrics = self.test(context)
            for name_metrics, metric in test_metrics.items():
                results[name_metrics] = metric

        return results, history

    def get_input_shape(self, train_gen):
        sample = train_gen[0][0]
        if isinstance(sample, list):
            input_shape = sample[0].shape[1:]
        else:
            input_shape = sample.shape[1:]
        return input_shape

    def test(self, context):
        model = context['model']
        test_gen = context['data_generators']['test']
        metrics = {}

        logging.info("Trainer :: Evaluating on test set")
        test_gen.batches_per_epoch = 100
        test_metrics = model.evaluate(test_gen, verbose=0)

        for i, metric_name in enumerate(model.metrics_names):
            # csv_config[f"test_{metric_name}"] = test_metrics[i]
            metrics[f"test_{metric_name}"] = test_metrics[i]
            logging.info(f"Trainer :: test_{metric_name}: {test_metrics[i]}")

        self.test_last_perentage(metrics, model, test_gen, 0.1)
        self.test_last_perentage(metrics, model, test_gen, 0.2)
        self.test_last_perentage(metrics, model, test_gen, 0.3)
        self.test_last_perentage(metrics, model, test_gen, 0)

        return metrics

    def test_last_perentage(self, metrics, model, test_gen, percentage):
        test_gen_ = test_gen.clone()
        test_gen_.ts_consider = percentage
        test_metrics = model.evaluate(test_gen_, verbose=0)
        for i, metric_name in enumerate(model.metrics_names):
            suffix = int(percentage*100) if percentage > 0 else "last"
            metrics[f"test{suffix}_{metric_name}"] = test_metrics[i]
