# CONELPABO

CONELPABO (Composite Networks Learning via Parallel Bayesian Optimization) is a framework designed to analyze long time-series data, with a focus on predicting the Remaining Useful Life (RUL) of systems and components. By using a divide-and-conquer strategy, the framework reduces hyperparameter search complexity and accelerates training by 50%, enabling the training of deeper networks even with limited computational resources.

Key Features:

 - **Efficient Model Training**: Parallel Bayesian Optimization and precomputed embeddings reduce training time and resource usage.
 - **State-of-the-Art Results**: Achieves superior performance on benchmark datasets, showcasing the effectiveness of CNN-CNN and RNN-RNN architectures.
 - **Reproducible Research**: Includes detailed parameter ranges and source code for replicating experiments.

## USAGE

The `search.py` script performs hyperparameter search for two neural network models: **FEN** (Feature Extractor Network) and **RNA MAS** (Remaining Useful Life Estimator). It leverages parallel Bayesian Optimization processes to optimize model configurations efficiently, reducing training time and resource usage.


```bash
python search.py -fm <fen_model> -rm <rna_model> -d <dataset> -o <output_dir> [options]
```

#### Required Arguments

| **Flag**   | **Long Name**   | **Description**                   | **Choices**              | **Type** |
|------------|-----------------|-----------------------------------|--------------------------|----------|
| `-fm`      | `--fen_model`   | Specifies the FEN model type      | `mscnn`, `rnn`, `mscnn1d` | `str`    |
| `-rm`      | `--rna_model`   | Specifies the RNA MAS model type. | `mscnn`, `rnn`, `mscnn1d` | `str`    |
| `-d`       | `--dataset`     | Path to the dataset metadata file. | -                        | `str`    |
| `-o`       | `--output`      | Directory to store the output and results. | -                        | `str`    |

#### Optional Arguments

| **Flag**   | **Long Name**    | **Description**                                   | **Choices** | **Default** | **Type** |
|------------|------------------|---------------------------------------------------|-------------|-------------|----------|
| `-c`       | `--cuda`         | Specifies the GPU to use (if available).          | `0`, `1`    | `""`        | `str`    |
| `-nc`      | `--ncpus`        | Number of CPU threads to use.                     | -           | `4`         | `int`    |
| `-ng`      | `--ngpus`        | Number of GPUs to use.                            | -           | `2`         | `int`    |
| `-b`       | `--debug`        | Enables debug mode for additional logging.        | -           | `False`     | `bool`   |

---

#### Examples

###### Basic Example
```bash
python search.py -fm mscnn -rm rnn -d PRONOSTIA  -o /path/to/output
```

###### Specify GPU
```bash
python search.py -fm mscnn1d -rm mscnn -d NCMPASS  -c 0 -o /path/to/output
```

###### Customize CPU and GPU Allocation
```bash
python search.py -fm rnn -rm rnn -d NCMPASS  -nc 8 -ng 1 -o /path/to/output
```

#### Outputs

###### Flags Directory
During execution, a directory named `flags` is created inside the specified output directory. This is used for synchronization between the two parallel Bayesian Optimization processes (`fen` and `rna`).

###### Results
Each process generates a file named `best_<model>_results.pk` in its output directory. These files contain the best hyperparameters and performance metrics for the respective models.


## ACKNOWLEDGMENT

This work has been supported by Grant PID2023-147198NB-I00 funded by MICIU/AEI/10.13039/501100011033  (Agencia Estatal de Investigación) and by FEDER, UE, and by the Ministry of Science and Education of Spain through the national program “Ayudas para contratos para la formación de investigadores en empresas (DIN2019-010887 / AEI / 10.13039/50110001103)”, of State Programme of Science Research and Innovations 2017-2020.
