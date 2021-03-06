#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:14:43 2018

@author: amrita
"""

import numpy as np
import random
import math
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n[p_index])

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n.sample_placeholder([None], name="action0")]

        p_input = obs_ph_n[0]

        p = p_func(p_input, int(act_pdtype_n.param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n.pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[0] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[0], act_input_n[0]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[0]], outputs=act_sample)
        p_values = U.function([obs_ph_n[0]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n.param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n.pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[0]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, u_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, u_estimation=False):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n[q_index])

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n.sample_placeholder([None], name="action0")]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        rew = tf.placeholder(tf.float32, [None], name="reward")
        if u_estimation:
            target_ph_u = tf.placeholder(tf.float32, [None], name="target_u")
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[0], act_ph_n[0]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        if u_estimation: 
            u_input = tf.concat(obs_ph_n + act_ph_n, 1)
            u = u_func(u_input, 1, scope="u_func", num_units=num_units)[:,0]
            u_loss = tf.reduce_mean(tf.square(tf.square(rew) + 2*tf.multiply(rew, target_ph) + target_ph_u - u))
            var = u - tf.square(q)
        else:
            var = tf.square(rew + target_ph) - tf.square(q)
        if u_estimation:
            u_func_vars = U.scope_vars(U.absolute_scope_name("u_func"))    
        q_loss = tf.reduce_mean(tf.square(q - (rew + target_ph)))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        if u_estimation:
            loss = q_loss + u_loss #+ 1e-3 * q_reg
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars+u_func_vars, grad_norm_clipping)
            train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [target_ph_u] + [rew], outputs=[q_loss, u_loss], updates=[optimize_expr])
            var_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [target_ph_u] + [rew], outputs=var)
        else:
            loss = q_loss #+ 1e-3 * q_reg
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
            train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=loss, updates=[optimize_expr])
            var_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=var)
        q_values = U.function(obs_ph_n + act_ph_n, q)
        
        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        
        if u_estimation:
            u_values = U.function(obs_ph_n + act_ph_n, u)
            target_u = u_func(u_input, 1, scope="target_u_func", num_units=num_units)[:,0]
            target_u_func_vars = U.scope_vars(U.absolute_scope_name("target_u_func"))
            update_target_u = make_update_exp(u_func_vars, target_u_func_vars)
            target_u_values = U.function(obs_ph_n + act_ph_n, target_u)

        if u_estimation:
            return train, update_target_q, update_target_u, {'q_values': q_values, 'u_values':u_values, 'var':var_fn, 'target_q_values': target_q_values, 'target_u_values': target_u_values}
        else:
            return train, update_target_q, {'q_values': q_values, 'var':var_fn, 'target_q_values': target_q_values}

class MADDPGAgentTrainerIndepLearner(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False, u_estimation=False):
        print ('in here')
        self.name = name
        self.n = 1#len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        obs_ph_n.append(U.BatchInput(obs_shape_n[agent_index], name="observation0").get())
        self.u_estimation = u_estimation
        
        # Create all the functions necessary to train the model
        l = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            u_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            u_estimation=self.u_estimation
        )
        
        if self.u_estimation:
            self.q_train, self.q_update, self.u_update, self.q_debug = l
        else:
            self.q_train, self.q_update, self.q_debug = l
            
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        obs_n = [obs]#+[np.zeros_like(obs)]*(self.n-1)
        obs_next_n = [obs_next]#+[np.zeros_like(obs_next)]*(self.n-1)
        act_n = [act]#+[np.zeros_like(act)]*(self.n-1)
            
        # train q network
        num_sample = 1
        target_q = 0.0
        target_u = 0.0
        for i in range(num_sample):
            t = self.p_debug['target_act'](obs_next_n[0])
            target_act_next_n = [t]# + [np.zeros_like(t)]*(self.n-1)
            #print('target_act_next_n ', np.asarray(target_act_next_n).shape)
            #print('obs_next_n', len(obs_next_n), obs_next_n[0].shape)
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            if self.u_estimation:
                target_u_next = self.q_debug['target_u_values'](*(obs_next_n + target_act_next_n))
                target_u += math.pow(self.args.gamma, 2.0) * (1.0 - done) * target_u_next
            target_q += self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        if self.u_estimation:
            q_loss, u_loss = self.q_train(*(obs_n + act_n + [target_q] + [target_u] + [rew]))
        else:
            q_loss = self.q_train(*(obs_n + act_n + [target_q] + [rew]))
        var_rew = np.array(self.q_debug['var'](*(obs_n + act_n + [target_q] + [rew]))).mean()
        # train p network
        p_loss = self.p_train(*(obs_n + act_n))
        
        self.p_update()
        self.q_update()
        if self.u_estimation:
            self.u_update()
            
        return [np.asarray(q_loss).mean(), np.asarray(p_loss).mean(), np.mean(target_q), np.mean(rew), var_rew, np.mean(target_q_next), np.std(target_q)]