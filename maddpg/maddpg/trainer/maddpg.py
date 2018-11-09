#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 00:19:22 2018

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
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, u_func, optimizer, optimizer_lamda, exp_var_alpha=None, cvar_alpha=None, cvar_beta=None, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, u_estimation=False, constrained=True, constraint_type=None, agent_type=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if constrained:
            lamda_constraint = tf.get_variable('lamda_constraint'+str(q_index), [1], initializer = tf.constant_initializer(1.0), dtype = tf.float32)
            if constraint_type=="CVAR":
                v_constraint = tf.get_variable('v_constraint'+str(q_index), [1], initializer = tf.constant_initializer(1.0), dtype = tf.float32)
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        if u_estimation:
            target_ph_u = tf.placeholder(tf.float32, [None], name="target_u")
        rew = tf.placeholder(tf.float32, [None], name="reward")
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        
        if u_estimation: 
            u_input = tf.concat(obs_ph_n + act_ph_n, 1)
            u = u_func(u_input, 1, scope="u_func", num_units=num_units)[:,0]
            u_loss = tf.reduce_mean(tf.square(tf.square(rew) + 2*tf.multiply(rew, target_ph) + target_ph_u - u))
            var = u - tf.square(q)
        else:
            var = tf.square(rew + target_ph) - tf.square(q)
        if constrained:    
            if constraint_type=="Exp_Var":
                #print ('In constraint generation with lamda alpha')
                constraint = lamda_constraint*(var - exp_var_alpha)
                q_loss = tf.reduce_mean(tf.square(q - (target_ph + rew - constraint)))
            elif constraint_type=="CVAR":
                cvar = v_constraint + (1.0/(1.0+cvar_alpha))*tf.reduce_mean(tf.nn.relu(q - v_constraint))
                if agent_type=="adversary":
                    constraint = lamda_constraint*(cvar - cvar_beta)
                elif agent_type=="good":
                    constraint = lamda_constraint*(cvar_beta - cvar)
                q_loss = tf.reduce_mean(tf.square(q - (target_ph + rew - constraint)))                      
        else:
            q_loss = tf.reduce_mean(tf.square(q - (target_ph + rew))) 
            
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        if u_estimation:
            u_func_vars = U.scope_vars(U.absolute_scope_name("u_func"))
        
        
        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        if u_estimation:
                loss = q_loss + u_loss #+ 1e-3 * q_reg
                optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars+u_func_vars, grad_norm_clipping)
                train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [target_ph_u] + [rew], outputs=[q_loss, u_loss], updates=[optimize_expr])
                var_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [target_ph_u] + [rew], outputs=var)
        elif constraint_type=="CVAR":
            loss = q_loss
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars + [v_constraint], grad_norm_clipping)
            train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=q_loss, updates=[optimize_expr])
            cvar_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=cvar)
            var_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=var)
        else:
            #print ('in loss minimization over q_func_vars')
            loss = q_loss
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
            train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=q_loss, updates=[optimize_expr])
            var_fn = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=var)
        #loss = loss + 1e-4*q_reg    
        # Create callable functions
        
        
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
            
        if constrained:   
            loss2  = -loss
            #print ('in loss maximisation over lamda')
            optimize_expr2 = U.minimize_and_clip(optimizer_lamda, loss2, [lamda_constraint], grad_norm_clipping)
            if u_estimation:
                train2 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [target_ph_u] + [rew], outputs=loss2, updates=[optimize_expr2])
            else:
                train2 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [rew], outputs=loss2, updates=[optimize_expr2])
                    
        if not u_estimation:    
            update_target_u = None
            target_u_values = None
            u_values = None
        if not constrained:
            train2 = None
            lamda_constraint = None
        if constraint_type!="CVAR":
            cvar_fn = None
            v_constraint = None
        return train, train2, update_target_q, update_target_u, {'q_values': q_values, 'u_values': u_values, 'target_q_values': target_q_values, 'target_u_values': target_u_values, 'var':var_fn, 'cvar':cvar_fn, 'lamda_constraint':lamda_constraint, 'v_constraint':v_constraint, 'optimize_expr':optimize_expr}
           
class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, agent_type, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.u_estimation = args.u_estimation
        self.constrained = args.constrained
        self.constraint_type = args.constraint_type
        self.agent_type = agent_type
        if self.agent_type=="good":
            cvar_alpha = args.cvar_alpha_good_agent
        elif self.agent_type=="adversary":
            cvar_alpha = args.cvar_alpha_adv_agent
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_train2, self.q_update, self.u_update, self.q_debug = q_train(
                scope=self.name,
                make_obs_ph_n=obs_ph_n,
                act_space_n=act_space_n,
                q_index=agent_index,
                q_func=model,
                u_func=model,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr_critic),
                optimizer_lamda=tf.train.AdamOptimizer(learning_rate=args.lr_lamda),
                exp_var_alpha=args.exp_var_alpha,
                cvar_alpha=cvar_alpha,
                cvar_beta=args.cvar_beta,
                grad_norm_clipping=0.5,
                local_q_func=local_q_func,
                num_units=args.num_units,
                u_estimation=self.u_estimation,
                constrained=self.constrained,
                constraint_type=self.constraint_type,
                agent_type=self.agent_type
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
        
    def clear_buffer(self):
        self.replay_buffer.clear()

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, frozen=False):
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
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        # train q network
        num_sample = 1
        target_q = 0.0
        if self.u_estimation:
            target_u = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]  
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q = self.args.gamma * (1.0 - done) * target_q_next
            if self.u_estimation:
                target_u_next = self.q_debug['target_u_values'](*(obs_next_n + target_act_next_n))
                target_u = math.pow(self.args.gamma, 2.0) * (1.0 - done) * target_u_next
        target_q /= num_sample
        if self.u_estimation:
            target_u/= num_sample
        if not frozen:    
            if self.u_estimation:
                q_loss, u_loss = self.q_train(*(obs_n + act_n + [target_q] + [target_u] + [rew]))
                if self.constrained:
                    q_loss2 = self.q_train2(*(obs_n + act_n + [target_q] + [target_u] + [rew]))
            else:
                q_loss = self.q_train(*(obs_n + act_n + [target_q] + [rew]))
                if self.constrained:
                    q_loss2 = self.q_train2(*(obs_n + act_n + [target_q] + [rew]))
            
            # train p network
            p_loss = self.p_train(*(obs_n + act_n))
    
            self.p_update()
            self.q_update()
            if self.u_estimation:
                self.u_update()
        if self.constrained:    
            lamda_constraint=np.array(self.q_debug['lamda_constraint'].eval()).mean()
            if lamda_constraint<=0:
                print("Value of Lamda violated",lamda_constraint)
        else:
            lamda_constraint = 0.0   
        if self.constraint_type=="CVAR":
            v_constraint=np.array(self.q_debug['v_constraint'].eval()).mean()
        else:
            v_constraint = 0.0
        if self.u_estimation:
            var_rew = np.array(self.q_debug['var'](*(obs_n + act_n + [target_q] + [target_u] + [rew]))).mean()
        else:    
            var_rew = np.array(self.q_debug['var'](*(obs_n + act_n + [target_q] + [rew]))).mean()
        if self.constrained and self.constraint_type=="CVAR":
            cvar = np.array(self.q_debug['cvar'](*(obs_n + act_n + [target_q] + [rew]))).mean()
        else:
            cvar = 0.0
        #tvars = tf.trainable_variables()
        #tvars_vals = tf.get_default_session().run(tvars)
        
        #for var, val in zip(tvars, tvars_vals):
        #    print("TF variable name",var.name)
        if not frozen:
            q_loss_mean = np.asarray(q_loss).mean()
            if self.u_estimation:
                u_loss_mean = np.asarray(u_loss).mean()
            else:
                u_loss_mean = 0.0
            p_loss_mean = np.asarray(p_loss).mean()
            if self.constrained:
                q_loss2_mean = np.asarray(q_loss2).mean()
            else:
                q_loss2_mean = 0.0
        else:
            q_loss_mean = 0.0
            u_loss_mean = 0.0
            p_loss_mean = 0.0
            q_loss2_mean = 0.0
        q_values = np.asarray(self.q_debug['q_values'](*(obs_n + act_n)))
        #print ('q_values', q_values.shape)
        mean_q_values = np.mean(q_values)
        std_q_values = np.std(q_values)
        return [q_loss_mean, u_loss_mean, q_loss2_mean, p_loss_mean, np.mean(rew), var_rew, cvar, lamda_constraint, v_constraint, mean_q_values, std_q_values]
         
