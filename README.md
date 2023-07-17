# Benchmarking LLM-based Task Planners

## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

1. Install and setup `git-lfs`.
    ```bash
    $ sudo apt-get install git-lfs
    $ git lfs install
    ```

2. Clone the whole repo.
    ```bash
    $ git clone {repo_url}
    ```

3. Setup a virtual environment.
    ```bash
    $ conda create -n {env_name} python=3.8
    $ conda activate {env_name}
    ```

4. Install PyTorch (2.0.0) first (see https://pytorch.org/get-started/locally/).
    ```bash
    # exemplary install command for PyTorch 2.0.0 with CUDA 11.7
    $ pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
    ```

5. Install python packages in `requirements.txt`.
    ```bash
    $ pip install -r requirements.txt
    ```

6. Download ALFRED dataset.
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


## FAQ

* Running out of disk space for Huggingface models
  * You can set the cache folder to be in another disk.
    ```bash
    $ export TRANSFORMERS_CACHE=/mnt/otherdisk/.hf_cache/
    ```

* I have encountered 'cannot find X server with xdpyinfo' in running ALFRED experiments.
  * Please try another x_display number (this should be a string; e.g., '1') in the config file.
    ```bash
    $ python evaluate.py --config-name=config_alfred alfred.x_display='1'
    ```


* Please use `startx.py` script to run ALFRED experiment on headless servers.

    ```bash
    $ sudo python3 alfred/scripts/startx.py 1
    ```
