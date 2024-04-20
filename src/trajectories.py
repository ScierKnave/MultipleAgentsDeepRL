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
    state = state.flatten()
    done = False
    states, actions_x, actions_y, rewards_x, rewards_y = [], [], [], [], []
    while not done:
        # action_x = torch.argmax(policy_x(state).sample().item())
        # action_y = torch.argmax(policy_y(state))
        state = torch.FloatTensor(state)
        action_x = policy_x(state).sample().item()
        action_y = policy_y(state).sample().item()
        state, reward_x, reward_y, done, _ = env.step(action_x, action_y)
        state = state.flatten()
        states.append(state)
        actions_x.append(action_x)
        actions_y.append(action_y)
        rewards_x.append(reward_x)
        rewards_y.append(reward_y)
    return torch.FloatTensor(states), torch.LongTensor(actions_x), torch.LongTensor(actions_y), torch.FloatTensor(rewards_x), torch.FloatTensor(rewards_y)
