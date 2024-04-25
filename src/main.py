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
from utils import *
from history_wrapper import *

def load_yaml_file(filepath):
    """Load a YAML file from the specified filepath."""
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def lola_training_loop(config, logger, env, policy_a, policy_b):

    for it in range(config['train_iterations']):

        nb = config['batch_size']
        trajectories = collect_trajectories(env=env, nb=nb, policy_x=policy_a, policy_y=policy_b)

        if (it+1) % config['log_freq'] == 0:

            print(f"Iteration: {it + 1}/{config['train_iterations']}")

    # Calculate the sum of each key in info dictionaries across all trajectories
        info_sums = {}
        info_counts = {}
        for info_list in trajectories['info']:
            for info in info_list:
                for key, value in info.items():
                    if key not in info_sums:
                        info_sums[key] = 0
                        info_counts[key] = 0.00001
                    if value is not None:
                        info_sums[key] += value
                        info_counts[key] += 1

        # Calculate and log the mean of each key's total sum
        info_means = {key: info_sums[key] / info_counts[key] for key in info_sums}
        for key, mean in info_means.items():
            logger.log(key, mean)
            print(f"{key} Mean: {mean:.2f}")

        logger.update_plots()
        logger.save_log()
        print("-" * 50)

        policy_a_copy = copy.deepcopy(policy_a)
        lola_pg_step(trajectories, policy_a, policy_b, config['lr'], config['lr_im'])
        switch_trajectories(trajectories) # sorry for this ugliness
        lola_pg_step(trajectories, policy_b, policy_a_copy, config['lr'], config['lr_im'])


def main():
    config = load_yaml_file('configs/config.yaml')


    if config['env'] == 'coin_game': env = RedBlueCoinGame(config['max_steps'])
    else: env = PrisonersDilemma(config['max_steps'])
    
    if config['history'] != "None":
        env = HistoryWrapper(env, config['history'])

    in_size = get_input_size(env.observation_space)

    out_size = env.action_space.n
    policy_a, policy_b = get_policies(in_size, out_size, config)

    logger = Logger(
                    config=config,
                    directory='experiments/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    k=config['video_freq']
                    )

    lola_training_loop(config, logger, env, policy_a, policy_b)


if __name__ == "__main__":
    main()


