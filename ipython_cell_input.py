
random = np.random.default_rng(2025)

_total_episodes_ = 10000

# Starting the SARSA learning

# init simbad
Q = init_simbad()

choice = []

for episode in range(_total_episodes_):
    #print('* episode', episode)
    env = init_environment()
    action = simbad_choose_action(env2state(env))
    #print(action)
    #render_environment(env)

    for i in range(_n_):
        #Getting the next state
        env2 = update_env(env, action)
 
        #Choosing the next action
        action2 = simbad_choose_action(env2state(env2))
        
        # getting reward for this next state
        reward = simbad_get_reward(env2state(env), action)
        
        if env.chosen:
            break
        
        #print('r', reward, action)
         
        #Learning the Q-value
        simbad_update(env2state(env), env2state(env2), reward, action)
        #plot_Q(Q) 

        env = env2
        action = action2
        #render_environment(env)
        
    choice.append(reward)
         
plot_Q(Q, kind='dual') 
plot_Q(Q, kind='diff') 
plot_Q(Q, kind='softmax') 
