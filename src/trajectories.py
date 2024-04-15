import torch

def switch_trajectories(trajectories):
    temp = trajectories['actions_x']
    trajectories['actions_x'] = trajectories['actions_y']
    trajectories['actions_y'] = temp

    temp = trajectories['rewards_x']
    trajectories['rewards_x'] = trajectories['rewards_y']
    trajectories['rewards_y'] = temp


def collect_trajectories(env, nb, policy_x, policy_y):
    trajectories = {}
    trajectories['states'] = []
    trajectories['actions_x'] = []
    trajectories['actions_y'] = []
    trajectories['rewards_x'] = []
    trajectories['rewards_y'] = []
    for _ in range(nb):
        states, actions_x, actions_y, rewards_x, rewards_y = collect_trajectory(env, policy_x, policy_y)
        trajectories['states'].append(states)
        trajectories['actions_x'].append(actions_x)
        trajectories['actions_y'].append(actions_y)
        trajectories['rewards_x'].append(rewards_x)
        trajectories['rewards_y'].append(rewards_y)
    return trajectories

def collect_trajectory(env, policy_x, policy_y):
    state = env.reset()
    done = False
    states, actions_x, actions_y, rewards_x, rewards_y = [], [], [], [], []
    while not done:
        action_x = torch.argmax(policy_x(state))
        action_y = torch.argmax(policy_y(state))
        state, reward_x, reward_y, done, _ = env.step(action_x, action_y)
        states.append(state)
        actions_x.append(action_x)
        actions_y.append(action_y)
        rewards_x.append(reward_x)
        rewards_y.append(reward_y)
    return states, actions_x, actions_y, rewards_x, rewards_y


def lola_pg(env, trajectories, policy_x, policy_y, policy_y_star):
    for trajectory in trajectories:
        states, actions_x, actions_y, rewards_x, rewards_y = trajectory

        lola_term = TODO
        #
        logits = policy_x(states)
        acs_dict = torch.distributions.Categoriacal(logits)
        loss =
