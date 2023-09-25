# Benchmarking LLM-based Task Planners

## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

1. Clone the whole repo.
    ```bash
    $ git clone {repo_url}
    ```

1. Setup a virtual environment.
    ```bash
    $ conda create -n {env_name} python=3.8
    $ conda activate {env_name}
    ```

1. Install PyTorch (2.0.0) first (see https://pytorch.org/get-started/locally/).
    ```bash
    # exemplary install command for PyTorch 2.0.0 with CUDA 11.7
    $ pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
    ```

1. Install python packages in `requirements.txt`.
    ```bash
    $ pip install -r requirements.txt
    ```

1. Download ALFRED dataset.
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
### Download the VirtualHome Simulator
- Download the VirtualHome simulator v2.2.2 and extract it
```bash
$ cd {project_root}/virtualhome/simulation/unity_simulator/
$ wget http://virtual-home.org//release/simulator/v2.0/v2.2.2/linux_exec.zip
$ unzip linux_exec.zip
```

### Benchmarking on Watch-And-Help-NL
- Open a new terminal and run VirtualHome simulator

```bash
$ cd {project_root}
$ ./virtualhome/simulation/unity_simulator/linux_exec.x86_64
```

- Open another terminal and evaluate.

```bash
$ cd {project_root}
$ python src/evaluate.py --config-name=config_wah
```

- You can override the configuration. We used [Hydra](https://hydra.cc/) for configuration management.

```bash
$ cd {project_root}
$ python evaluate.py --config-name=config_wah planner.model_name=EleutherAI/gpt-neo-1.3B planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=10 
```

### Benchmarking on Watch-And-Help-NL Using Headless PC
- Open a new terminal and run Xserver
```bash
$ cd {project}/virtualhome
$ sudo python helper_scripts/startx.py $display_num
```
- Open another terminal and run unity simulator
```bash
$ cd {project}/virtualhome
$ DISPLAY=:$display_num ./simulation/unity_simulator/linux_exec.x86_64 -batchmode
```
- Open another terminal and evaluate
```bash
$ cd {project_root}
$ python src/evaluate.py --config-name=config_wah_headless
```


## Extensions

### In-context example selection
```bash
$ python src/evaluate.py --config-name=config_wah prompt.select_method=same_task
$ python src/evaluate.py --config-name=config_wah prompt.select_method=topk
```

### Replanning
```bash
$ python src/evaluate.py --config-name=config_alfred planner.use_predefined_prompt=True
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
