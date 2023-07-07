# Benchmarking LLM-based Task Planners

## Install

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

We recommends using a virtual environment.

```console
$ conda create -n {env_name} python=3.8
$ conda activate {env_name}
```
Install PyTorch (>=1.11.0) first (see https://pytorch.org/get-started/locally/)
then install python packages in `requirements.txt`

```console
$ pip install -r requirements.txt
```


## Benchmarking on ALFRED

```console
$ python evaluate.py
```

You can override the configuration. We used [Hydra](https://hydra.cc/) for configuration managing.

```console
$ python evaluate.py planner.model="EleutherAI/gpt-neo-125M"
```


## Benchmarking on Watch-And-Help
```console
$ cd {project_root}
$ ./script/icra_exp1_benchmark_wah.sh
```
