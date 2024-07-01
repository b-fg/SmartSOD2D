import random, os, numpy as np

f_control_classic = 0.0025
t_control_classic = 1.0 / f_control_classic
t_action = 0.1
t_episode_train = 1.0
t_episode_eval = 5.0
t_begin_control = 0.1 # controls begin after this value
cfd_n_envs = 2
marl_n_envs = 3
mode = "train" # "train" or "eval"

params = {
    # smartsim params
    "port": random.randint(6000, 7000), # generate a random port number
    "num_dbs": 1,
    "network_interface": "ib0", # "lo", "ib0"
    "run_command": "mpirun",
    "launcher": "slurm", # "local", "slurm", "slurm-split"
    "cluster_account": "NAISS2023-5-102", # for slurm_split usage, as in Alvis
    "sod2d_modules": "/mimer/NOBACKUP/groups/deepmechalvis/bernat/smartsod2d/utils/modules-alvis-nvhpc.sh", # for slurm-split use cases
    "episode_walltime": "02:00:00",

    # environment params
    "cfd_n_envs": cfd_n_envs,
    "marl_n_envs": marl_n_envs,
    "n_envs": cfd_n_envs * marl_n_envs,
    "n_tasks_per_env": 1,
    "witness_file": "witness.txt",
    "witness_xyz": (6, 6, 6),
    "marl_neighbors": 1, # 0 is local state only
    "rectangle_file": "rectangleControl.txt",
    "time_key": "time",
    "step_type_key": "step_type",
    "state_key": "state",
    "state_size_key": "state_size",
    "action_key": "action",
    "action_size_key": "action_size",
    "reward_key": "reward",
    "dtype": np.float32,
    "sod_dtype": np.float64,
    "poll_time": 360000,
    "verbosity": "debug", # quiet, debug, info
    "dump_data_flag": True,

    # RL params
    "mode": mode,
    "num_episodes": 10, # total MARL episodes
    "num_epochs": cfd_n_envs * marl_n_envs, # number of epochs to perform policy (optimizer) update per episode sampled. Rule of thumb: n_envs.
    "f_control_classic": f_control_classic,
    "t_action": t_action,
    "f_action": 1.0 / t_action,
    "t_episode": t_episode_train if mode == "train" else t_episode_eval,
    "t_begin_control": t_begin_control,
    "action_bounds": (-0.3, 0.3),
    "reward_norm": 153.6, # non-actuated lx in coarse mesh
    "reward_beta": 0.5, # reward = beta * reward_global + (1.0 - beta) * reward_local
    "restart_file": 3, # 3: random. 1: restart 1. 2: restart 2
    "net": (128, 128),
    "learning_rate": 5e-4,
    "replay_buffer_capacity": int(t_episode_train / t_action) + 1, # trajectories buffer
    "log_interval": 1, # save model, policy, metrics, interval
    "summary_interval": 1, # write to tensorboard interval
    "seed": 16,
    "ckpt_num": int(1e6),
    "ckpt_interval": 1,
    "use_XLA": True,
    "do_profile": False,
    "use_tf_functions": True
}

# Default params
params["collect_episodes_per_iteration"] = params["n_envs"] # number of episodes to collect before each optimizer update
os.environ["SMARTSIM_LOG_LEVEL"] = params["verbosity"] # quiet, info, debug
os.environ["SR_LOG_LEVEL"] = params["verbosity"] # quiet, info, debug
os.environ["SR_LOG_FILE"] = "sod2d_exp/sr_log_file.out" # SR output log

# Params groups
env_params = {
    "launcher": params["launcher"],
    "run_command": params["run_command"],
    "cluster_account": params["cluster_account"],
    "sod2d_modules": params["sod2d_modules"],
    "episode_walltime": params["episode_walltime"],
    "n_tasks_per_env": params["n_tasks_per_env"],
    "marl_n_envs": params["marl_n_envs"],
    "f_action": params["f_action"],
    "t_episode": params["t_episode"],
    "t_begin_control": params["t_begin_control"],
    "action_bounds": params["action_bounds"],
    "reward_norm": params["reward_norm"],
    "reward_beta": params["reward_beta"],
    "dtype": params["dtype"],
    "sod_dtype": params["sod_dtype"],
    "poll_time": params["poll_time"],
    "witness_file": params["witness_file"],
    "witness_xyz": params["witness_xyz"],
    "rectangle_file": params["rectangle_file"],
    "time_key": params["time_key"],
    "step_type_key": params["step_type_key"],
    "state_key": params["state_key"],
    "state_size_key": params["state_size_key"],
    "action_key": params["action_key"],
    "action_size_key": params["action_size_key"],
    "reward_key": params["reward_key"],
    "dump_data_flag": params["dump_data_flag"],
}
