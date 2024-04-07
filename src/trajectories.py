import torch

def collect_trajectory(env, policy_x, policy_y):
    state = env.reset()
    done = False
    states, actions_x, actions_y, rewards_x, rewards_y = [], [], [], [], []
    while not done:
        action_x = torch.argmax(policy_x(state))
        action_y = torch.argmax(policy_y(state))
        state, reward_x, reward_y, done, _ = env.step(action_x, action_y)
        states.append[state]
        actions_x.append[action_x] 
        actions_y.append[action_y] 
        rewards_x.append[reward_x]
        rewards_y.append[reward_y]
    return states, actions_x, actions_y, rewards_x, rewards_y


def lola_pg(env, trajectories, policy_x, policy_y, policy_y_star):
    for trajectory in trajectories:
        states, actions_x, actions_y, rewards_x, rewards_y = trajectory

        lola_term = TODO
        # 
        logits = policy_x(states)
        acs_dict = torch.distributions.Categoriacal(logits)
        loss = 


    