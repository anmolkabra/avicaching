# GPU-accelerated Principal-Agent Game for Scalable Citizen Science

Efficiently reducing sampling bias in citizen science programs using principal-agent games, such as _Avicaching_ in [_eBird_](https://ebird.org/home), a bird-observational dataset.

**Authors:** [Anmol Kabra](https://anmolkabra.com/), [Yexiang Xue](https://www.cs.purdue.edu/homes/yexiang/), [Carla P. Gomes](https://www.cs.cornell.edu/gomes/).

The publication is available at [anmolkabra.com/docs/avicaching-compass19.pdf](https://anmolkabra.com/docs/avicaching-compass19.pdf) ([doi: 10.1145/3314344.3332495](https://doi.org/10.1145/3314344.3332495)), and this work is licensed CC-BY-4.0.

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

If you find this work useful, please cite it as:

```
Anmol Kabra, Yexiang Xue, and Carla P. Gomes. 2019. GPU-accelerated Principal-Agent Game for Scalable Citizen Science.
In ACM SIGCAS Conference on Computing and Sustainable Societies (COMPASS) (COMPASS ’19), July 3–5, 2019, Accra, Ghana.
ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3314344.3332495
```

[update bibtex citation]

## Abstract

> Citizen science programs have been instrumental in boosting sustainability projects, large-scale scientific discovery, and crowdsourced experimentation.
Nevertheless, these programs witness challenges in submissions' quality, such as sampling bias resulting from citizens' preferences to complete some tasks over others.
The sampling bias frequently manifests itself in the program's dataset as spatially clustered submissions, which reduce the efficacy of the dataset for subsequent scientific studies.
To address the spatial clustering problem, programs use reward schemes obtained from game-theoretical models to incentivize citizens to perform tasks that are more meaningful from a scientific point of view.
Herein we propose a GPU-accelerated approach for the _Avicaching_ game, which was recently introduced by the _eBird_ citizen science program to incentivize birdwatchers to collect bird data from under-sampled locations.
_Avicaching_ is a Principal-Agent game, in which the principal corresponds to the citizen science program (_eBird_) and the agents to the birdwatchers or citizen scientists.
Previous approaches for solving the _Avicaching_ game used approximations based on mixed-integer programming and knapsack algorithms combined with learning algorithms, using standard CPU hardware.
Following the recent advances in scalable deep learning and parallel computation on Graphical Processing Units (GPUs), we propose a novel approach to solve the _Avicaching_ game, which takes advantage of neural networks and parallelism for large-scale games.
We demonstrate that our approach better captures agents' behavior, which allows better learning and more effective incentive distribution in a real-world bird observation dataset.
Our approach also allows for massive speedups using GPUs.
As _Avicaching_ is representative of games that are aimed at reducing spatial clustering in citizen science programs, our scalable reformulation for _Avicaching_ enables citizen science programs to tackle sampling bias and improve submission quality on a large scale.

## Installation

The project is tested in Ubuntu 16.04 64-bit, though we believe it would work in any Linux 64-bit OS.

Clone the repository and install the conda environment `avicaching` from `environment.yml` file as:

```bash
conda env create -f environment.yml
```

You can change the name of the conda environment by modifying the first line of the `environment.yml` file.

## Data

We provide synthetic datasets for setup purposes and running scalability experiments.
Please email [ak2426@cornell.edu](mailto:ak2426@cornell.edu?Subject=[Avicaching]) if you need access to the original _eBird_ data or other files used for our experiments.

## Usage

1. The outputs of the scripts require this directory structure:
    ```
    - stats/
        - find_weights/
            - logs/
            - map_plots/
            - plots/
            - weights/
        - find_rewards/
            - logs/
            - plots/
            - test_rewards_results/
    ```
    You can create this structure with:
    ```bash
    for dir in logs map_plots plots weights; do mkdir -p "stats/find_weights/$dir/"; done
    for dir in logs plots test_rewards_results; do mkdir -p "stats/find_rewards/$dir"; done
    ```
2. Running the `nn_avicaching_find_weights.py` file will run the identification problem models.
You will have to specify the number of layers in the model with flag `--layers k`.
3. Running the `nn_avicaching_find_rewards.py` file with the location of the weights files from identification problem models (specified with `--weights-file filename`) will run the pricing problem models.
The script will automatically set the number of layers to the one used in the identification problem model.
4. All other flags in both scripts are optional, as they are default set to the basic options.
