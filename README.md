# Pseudopod Competition as a decision-making mechanism
**Code for the paper on pseudopod splitting during chemotaxis.**

## Project structure
The project is structured as follows:

```
├── README.md 
├── psxc/ # Folder containing the code related to the dynamical model and RL training.
├── scripts/ # Folder containing scripts to evaluate and create figures.
├── pyproject.toml  # Package setup for psxc (needed to run scripts)
└── requirements.txt # Project dependencies
```

# Setup
Create a virtual environment and install the requiered packages (Note: we use `jaxlib` for GPU)

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you need to run it on a GPU, you can install jax as
```
pip install jax[cuda12]
```

Likewise, you need to install this library locally in order to run the scripts

```
pip install -e .
```
