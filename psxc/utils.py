import pathlib
import orbax.checkpoint as ocp

def load_last_model(path):
    runs_dir = pathlib.Path(path)
    last_run = sorted(runs_dir.glob("ppo_discrete_*"), key=lambda x: x.stat().st_ctime)[-1]
    print(f'Loading model from {last_run}')
    checkpointer = ocp.StandardCheckpointer()
    discrete = last_run.stem.split('_')[1] == 'discrete'
    return checkpointer.restore((last_run/"checkpoint").absolute()), discrete

