#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:55:47 2018

@author: amrita
"""

import numpy as np
import random
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

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, optimizer_lamda, alpha=None, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        lamda_constraint = tf.get_variable('lamda_constraint'+str(q_index), [1], initializer = tf.constant_initializer(1.0), dtype = tf.float32)
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n[q_index])
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n.sample_placeholder([None], name="action0")]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        rew = tf.placeholder(tf.float32, [None], name="reward")
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[0], act_ph_n[0]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        var = tf.square(rew+target_ph) - tf.square(q)
        constraint = rew - lamda_constraint*(var - alpha)
        target = target_ph + constraint
        
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        q_loss = tf.reduce_mean(tf.square(q - target)) #+ lamda_constraint*(var_rew - alpha)
        
        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        x = obs_ph_n + act_ph_n + [target_ph] +[rew]
        train = U.function(inputs=x, outputs=loss, updates=[optimize_expr])
        var_fn = U.function(inputs=x, outputs=var)
        
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        loss2  = -q_loss
        optimize_expr2 = U.minimize_and_clip(optimizer_lamda, loss2, [lamda_constraint], grad_norm_clipping)
        x2 = obs_ph_n + act_ph_n + [target_ph] +[rew]
        train2 = U.function(inputs=x2, outputs=loss2, updates=[optimize_expr2])

        
        return train, train2, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values, 'var':var_fn, 'lamda_constraint':lamda_constraint,'optimize_expr':optimize_expr}

class MADDPGAgentTrainerIndepLearner(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        print ('In MADDPGAgentTrainerIndepLearner')
        self.name = name
        self.n = 1
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        obs_ph_n.append(U.BatchInput(obs_shape_n[agent_index], name="observation0").get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_train2, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr_critic),
            optimizer_lamda=tf.train.AdamOptimizer(learning_rate=args.lr_lamda),
            alpha=args.alpha,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr_actor),
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
        obs_n.append(obs)
        obs_next_n.append(obs_next)
        act_n.append(act)
        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [self.p_debug['target_act'](obs_next_n[0])]  # WHY IS THIS ON AGENT[0]'s target_act ????????
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q = self.args.gamma * (1.0 - done) * target_q_next
            #rew += (rew - self.lamda_constraint*(var_rew - self.args.alpha))
        target_q /= num_sample
        
        q_loss = self.q_train(*(obs_n + act_n + [target_q] + [rew]))
        q_loss2 = self.q_train2(*(obs_n + act_n + [target_q] + [rew]))
        
        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()
        #tvars = tf.trainable_variables()
        #tvars_vals = tf.get_default_session().run(tvars)
        
        #for var, val in zip(tvars, tvars_vals):
        #    print("TF variable name",var.name) 
        lamda_constraint=np.array(self.q_debug['lamda_constraint'].eval()).mean()
        var_rew = np.array(self.q_debug['var'](*(obs_n + act_n + [target_q] + [rew]))).mean()
        if lamda_constraint<=0:
            print("Value of Lamda violated",lamda_constraint)
        return [np.asarray(q_loss).mean(), np.asarray(q_loss2).mean(), np.asarray(p_loss).mean(), np.mean(target_q), np.mean(rew), var_rew, np.mean(target_q_next), np.std(target_q), lamda_constraint]

