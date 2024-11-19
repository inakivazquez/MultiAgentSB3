import json
import os
import shutil
import json


import gymnasium as gym
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from ma_sb3 import TimeLimitMAEnv

from ma_sb3.envs.soccer_v0 import SoccerEnv

# Define the hyperparameters to optimize
hyperparameters = {
    'learning_rate': (1e-4, 1e-2),
    'gamma': (0.9, 0.99),
    'batch_size': [32, 64, 128, 256],
    'use_sde': (True, False),    
    'net_arch_nodes': [32, 64, 128, 256],    
}

# Define the objective function for Optuna
def objective(trial: optuna.Trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', *hyperparameters['learning_rate'], log=True)
    gamma = trial.suggest_float('gamma', *hyperparameters['gamma'])
    batch_size = trial.suggest_categorical('batch_size', hyperparameters['batch_size'])
    use_sde = trial.suggest_categorical('use_sde', hyperparameters['use_sde'])
    net_arch_nodes = trial.suggest_categorical('net_arch_nodes', hyperparameters['net_arch_nodes']) 

    policy_kwargs = dict(net_arch=[net_arch_nodes, net_arch_nodes])
    seed = 42

    model = PPO('MlpPolicy', env, seed=seed, verbose=True,  gamma=gamma,  learning_rate=learning_rate,
                        batch_size=batch_size,  use_sde=use_sde, policy_kwargs=policy_kwargs)    
    
    print(f"Trial {trial.number} with hyperparameters: {trial.params}")

    try:
        model.learn(total_timesteps=n_steps, progress_bar=True)
        model.save(f"{full_study_dir_path}/trial_{trial.number}.zip")
        print()
        print("Evaluating the model...")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print("Mean reward:", mean_reward)
        return mean_reward
    except Exception as e: # Sometimes learn can fail due to exploding gradients
        print("Skipping trial due to an error:")
        print(e)
        return float('-inf')


def create_study_dir(optuna_dir, study_dir, delete_existing=True):
    # Create the first level directory if it does not exist
    if not os.path.exists(optuna_dir):
        os.makedirs(optuna_dir)
        print(f"Directory: {optuna_dir} created.")
    
    # Full path for the study directory
    full_study_dir_path = os.path.join(optuna_dir, study_dir)
    
    # If the second level directory exists, remove it and its contents
    if os.path.exists(full_study_dir_path):
        if delete_existing:
            shutil.rmtree(full_study_dir_path)
            print(f"Removed existing study directory and all its contents: {full_study_dir_path}")

    # Create the second level directory if required
    os.makedirs(full_study_dir_path, exist_ok=True)
    print(f"Created study directory: {full_study_dir_path}")
    os.makedirs(os.path.join(full_study_dir_path, "models"), exist_ok=True)


# Create environment
n_players_per_team = 1
single_team = True
experiment_name = "single_soccer_ppo_easygoal"

env_params = {'n_team_players': n_players_per_team, 'single_team': single_team, 'perimeter_side': 10}

ma_env = SoccerEnv(**env_params, render=False)
ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

agents_envs = ma_env.get_agents_envs()

env = agents_envs['red_0']
storage_file = f"sqlite:///optuna/optuna.db"
study_name = experiment_name
full_study_dir_path = f"optuna/{study_name}"

study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)
n_trials = 50
n_steps = 200_000

print(f"Searching for the best hyperparameters in {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

env.close()

# Create the study directory if required
create_study_dir("optuna", study_name, delete_existing=True)

best_trial = study.best_trial

# Generate the policy_kwargs key before writing to file
net_arch_nodes = best_trial.params.pop('net_arch_nodes')
best_trial.params['policy_kwargs'] = {'net_arch': [net_arch_nodes, net_arch_nodes]}
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)

# save the data in a JSON file
best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
best_trial_file.write(best_trial_params)
best_trial_file.close()


# Generate the figures of the results
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(f"{full_study_dir_path}/optimization_history.html")
fig = optuna.visualization.plot_contour(study)
fig.write_html(f"{full_study_dir_path}/contour.html")
fig = optuna.visualization.plot_slice(study)
fig.write_html(f"{full_study_dir_path}/slice.html")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(f"{full_study_dir_path}/param_importances.html")

