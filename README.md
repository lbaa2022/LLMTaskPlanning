# Benchmarking LLM-based Task Planners

## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

We recommends using a virtual environment.

```bash
$ conda create -n {env_name} python=3.8
$ conda activate {env_name}
```

Install PyTorch (>=1.11.0) first (see https://pytorch.org/get-started/locally/).

```bash
# exemplary command for PyTorch 1.13.0 with CUDA 11.6
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, install python packages in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

Download ALFRED dataset.

```bash
$ cd alfred/data
$ sh download_data.sh json
```


## Benchmarking on ALFRED

```bash
$ python src/evaluate.py --config-name=config_alfred
```

You can override the configuration. We used [Hydra](https://hydra.cc/) for configuration management.

```bash
$ python evaluate.py --config-name=config_alfred planner.model=EleutherAI/gpt-neo-125M
$ python evaluate.py --config-name=config_alfred alfred.x_display='1'
```


## Benchmarking on Watch-And-Help
```bash
$ cd {project_root}
$ ./script/icra_exp1_benchmark_wah.sh
```


## Extract train samples from ALFRED for language model finetuning

Make sure you have preprocessed data (run ALFRED benchmarking at least once).

```bash
$ python src/misc/extract_alfred_train_samples.py
```


## Tips

* If you're running out of disk space for Huggingface models, you can set the cache folder to be in another disk.

```bash
$ export TRANSFORMERS_CACHE=/mnt/otherdisk/.hf_cache/
```

* Please use `startx.py` script to run ALFRED experiment on headless servers.

```bash
$ sudo python3 alfred/scripts/startx.py 1
```
