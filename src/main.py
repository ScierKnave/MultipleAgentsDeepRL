import yaml
from trajectories import *
from policies import *
from lola import *
from coin_game import *
from prisonner_dilemma import *
import json
from datetime import datetime
import statistics as st
import coin_game
import prisonner_dilemma
from utils import get_input_size

def load_yaml_file(filepath):
    """Load a YAML file from the specified filepath."""
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def output_log(log_dict, file_path='log.json'):
    """Saves the log dictionary to a file in JSON format."""
    with open(file_path, 'w') as file:
        json.dump(log_dict, file, indent=4)
    print(f"Log saved to {file_path}")

def lola_training_loop(config, env, policy_a, policy_b):
    logger = {}
    logger['rewards_a'] = []
    logger['rewards_b'] = []

    for it in range(config['train_iterations']):

        nb = config['batch_size']
        trajectories = collect_trajectories(env=env, nb=nb, policy_x=policy_a, policy_y=policy_b)

        if (it + 1) % config['log_freq'] == 0:
            mean_reward_a = st.mean([rewards.sum().item() for rewards in trajectories['rewards_x']])
            mean_reward_b = st.mean([rewards.sum().item() for rewards in trajectories['rewards_y']])
            logger['rewards_a'].append(mean_reward_a)
            logger['rewards_b'].append(mean_reward_b)
            print(f"Iteration: {it + 1}/{config['train_iterations']}")
            print(f"Mean Reward A: {mean_reward_a:.2f}")
            print(f"Mean Reward B: {mean_reward_b:.2f}")
            print("-" * 50)

        lola_pg_step(trajectories, policy_a, policy_b, config['lr'], config['lr_im'])

        switch_trajectories(trajectories) # sorry for this ugliness
        lola_pg_step(trajectories, policy_b, policy_a, config['lr'], config['lr_im'])

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_log(logger, file_path=f"training_log_{current_time}.json")



def main():
    # Example: Loading a YAML configuration file
    config = load_yaml_file('configs/config.yaml')

    if config['env'] == 'coin_game': env = RedBlueCoinGame(config['max_steps'])
    else: env = PrisonersDilemma(config['max_steps'])
    # else: env = CoinGame(config['max_steps'])

    in_size = get_input_size(env.observation_space)

    out_size = env.action_space.n
    policy_a, policy_b = get_policies(in_size, out_size, config)

    lola_training_loop(config, env, policy_a, policy_b)


if __name__ == "__main__":
    main()
