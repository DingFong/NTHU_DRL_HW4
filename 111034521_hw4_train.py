from osim.env import L2M2019Env
from model.agent import Agent
from env_wrapper import wrapper

import itertools

import argparse
from torch.utils.tensorboard import SummaryWriter




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'Discount factor')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--memory_capacity', type = int, default= 4000000)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--n_iteractions', type = int, default = 800000)
    parser.add_argument('--eval', type = bool, default = True)
    parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
    
    # for exploration
    parser.add_argument('--exploration', type = bool, default = False)
    parser.add_argument('--n_state_predictor', type = int, default = 5)
    parser.add_argument('--hidden_sizes', type = list, default = [64, 64])
    parser.add_argument('--lr_exploration', type = float, default = 0.001)
    parser.add_argument('--bonus_scale', type = float, default = 1)

    
    args = parser.parse_args()

    dir = 'runs/SAC_AET({})_niteractions({})_exploration:({})'.format(args.automatic_entropy_tuning, args.n_iteractions, args.exploration)
    print(dir)
    writer = SummaryWriter(dir)
    # writer = SummaryWriter('runs/test')

    env = L2M2019Env(difficulty=2, visualize = False)
    env = wrapper(env)
    num_inputs = env.observation_space.shape[0]
    
    
    n_interactions = args.n_iteractions
    agent = Agent(env.observation_space, env.action_space, args)

    Total_steps = 0
    updates = 0

    best_test_reward = -10000
    for i_episode in itertools.count(1):
        cur_state = env.reset()
        
        episode_reward = 0
        episode_steps = 0

        while True:
            
            action = agent.select_action(cur_state)
            
            next_state, reward, done, info = env.step(action.flatten())
            # print(next_state)
            # exploration
            
            r_bonus = agent.exploration.get_exploration_bonus(cur_state, action.flatten()) if args.exploration else 0
            reward += r_bonus
            
            # print('reward: ', reward)

            mask = 1 - done
            agent.update_memory(cur_state, action, reward, next_state, mask)
            
            if len(agent.memory) > args.batch_size:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_params(updates)
                writer.add_scalar('loss/critic_1_loss', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2_loss', critic_2_loss, updates)
                writer.add_scalar('loss/policy_loss', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('alpha', alpha, updates)

                if args.exploration:
                    exploration_predictor_loss = agent.train_exploration()
                    writer.add_scalar('loss/exploration_predictor_loss', exploration_predictor_loss, updates)
                
                updates += 1

            
            episode_reward += reward
            episode_steps += 1

            cur_state = next_state

            if done:
                break
        
        
        # check num of steps if exceeding n_interactions
        Total_steps += episode_steps
        writer.add_scalar('reward/train', episode_reward, i_episode)
        writer.add_scalar('Memory size', len(agent.memory), i_episode)


        if Total_steps > n_interactions:
            break

        if i_episode % 10 == 0 and args.eval is True:
            test_avg_reward = 0.
            test_episodes = 10
            for _  in range(test_episodes):
                test_state = env.reset()
                test_episode_reward = 0
                test_done = False
                while not test_done:
                    test_action = agent.select_action(test_state, evaluate = True)

                    test_next_state, test_reward, test_done, _ = env.step(test_action)
                    test_episode_reward += test_reward


                    test_state = test_next_state
                test_avg_reward += test_episode_reward

            
                    
            test_avg_reward /= test_episodes

            if test_avg_reward > best_test_reward:
                best_test_reward = test_avg_reward
                agent.save_checkpoint(i_episode)

            writer.add_scalar('reward/test', test_avg_reward, i_episode)

