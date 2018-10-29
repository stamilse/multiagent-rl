import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import math
sys.path.append('..')
sys.path.append('../../multiagent-particle-envs')
import multiagent
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.maddpg_indep_learner import MADDPGAgentTrainerIndepLearner
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--independent-learner", type=str, default="False", help="use independent learner or not")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=0.5*1e-2, help="learning rate for the actor for Adam optimizer")
    parser.add_argument("--lr_critic", type=float, default=1e-2, help="learning rate for the critic for Adam optimizer")
    parser.add_argument("--lr_lamda", type=float, default=1e-4, help="learning rate for the lamda update using Adam optimizer")
    parser.add_argument("--u_estimation", type=str, default="False", help="explicitly learn u-values for better estimation of variance")
    parser.add_argument("--constrained", type=str, default="True", help="objective is a constrained minimization or not")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--alpha", type=float, default=0.005, help="alpha for controlling the variance constraint")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_agents_inpt=2*arglist.num_adversaries, num_adversaries_inpt=arglist.num_adversaries)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    if arglist.independent_learner:
        trainer = MADDPGAgentTrainerIndepLearner
    else:    
        trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'), u_estimation=arglist.u_estimation, constrained=arglist.constrained))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg'), u_estimation=arglist.u_estimation, constrained=arglist.constrained))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        episode_variance = [0.0]
        agent_variance = [[0.0] for _ in range(env.n)] 
        final_ep_variance = []
        final_ep_ag_variance = []
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        avg_q_loss = [[] for i in range(env.n)]
        if arglist.u_estimation:
            avg_u_loss = [[] for i in range(env.n)]
        avg_p_loss = [[] for i in range(env.n)]
        avg_mean_rew = [[] for i in range(env.n)]
        avg_var_rew = [[] for i in range(env.n)]
        avg_lamda = [[] for i in range(env.n)]
        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                episode_variance.append(0)
                for a in agent_rewards:
                    a.append(0)
                for a in agent_variance:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for index,agent in enumerate(trainers):
                returned = agent.update(trainers, train_step)
                if returned is None:
                    continue
                if arglist.u_estimation:
                    q_loss, u_loss, p_loss, mean_rew, var_rew, lamda = returned
                else:
                    q_loss, p_loss, mean_rew, var_rew, lamda = returned
                episode_variance[-1] += max(0,var_rew)
                agent_variance[index][-1] += max(0,var_rew) 
                #print ('qloss value::::::::::::',q_loss)
                avg_q_loss[index].append(q_loss)
                if arglist.u_estimation:
                    avg_u_loss[index].append(u_loss)
                avg_p_loss[index].append(p_loss)
                avg_mean_rew[index].append(mean_rew)
                avg_var_rew[index].append(max(0,var_rew))
                avg_lamda[index].append(lamda)
            # save model, display training output
            if terminal and len(episode_rewards)>0 and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("steps: {}, episodes: {}, mean episode variance: {}, time: {}".format(
                        train_step, len(episode_variance), np.mean(episode_variance[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                    print("steps: {}, episodes: {}, mean episode variance: {}, agent episode variance: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_variance[-arglist.save_rate:]),
                        [np.mean(var[-arglist.save_rate:]) for var in agent_variance], round(time.time()-t_start, 3)))
                for agent_index in range(env.n):
                    avg_q_loss[agent_index] = np.asarray(avg_q_loss[agent_index])
                    if arglist.u_estimation:
                        avg_u_loss[agent_index] = np.asarray(avg_u_loss[agent_index])
                    avg_p_loss[agent_index] = np.asarray(avg_p_loss[agent_index])
                    avg_mean_rew[agent_index] = np.asarray(avg_mean_rew[agent_index])
                    avg_var_rew[agent_index] = np.asarray(avg_var_rew[agent_index])
                    avg_lamda[agent_index] = np.asarray(avg_lamda[agent_index])
                    if arglist.u_estimation:
                        print('Running avgs for agent {}: q_loss: {}, u_loss: {}, p_loss: {}, mean_rew: {}, variance: {}, lamda: {}'.format(
                            agent_index, np.mean(avg_q_loss[agent_index]), np.mean(avg_u_loss[agent_index]), 
                            np.mean(avg_p_loss[agent_index]), np.mean(avg_mean_rew[agent_index]), np.mean(avg_var_rew[agent_index]), np.mean(avg_lamda[agent_index])))
                    else:
                        print('Running avgs for agent {}: q_loss: {}, p_loss: {}, mean_rew: {}, var_rew: {}, lamda: {}'.format(
                            agent_index, np.mean(avg_q_loss[agent_index]), np.mean(avg_p_loss[agent_index]), 
                            np.mean(avg_mean_rew[agent_index]), np.mean(avg_var_rew[agent_index]), np.mean(avg_lamda[agent_index])))
                    avg_q_loss[agent_index] = []
                    if arglist.u_estimation:
                        avg_u_loss[agent_index] = []
                    avg_p_loss[agent_index] = []
                    avg_mean_rew[agent_index] = []
                    avg_var_rew[agent_index] = []
                    avg_lamda[agent_index] = []
                    print ('')
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                final_ep_variance.append(np.mean(episode_variance[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                for var in agent_variance:    
                    final_ep_ag_variance.append(np.mean(var[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    if arglist.independent_learner=="True":
        arglist.independent_learner = True
    else:
        arglist.independent_learner = False
    if arglist.u_estimation=="True":
        arglist.u_estimation = True
    else:
        arglist.u_estimation = False
    if arglist.constrained =="True":
        arglist.constrained = True
    else:
        arglist.constrained = False
    print ('arglist.u_estimation',arglist.u_estimation)
    train(arglist)
