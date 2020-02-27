import gym
import numpy as np
from gym_monopoly import envs

def simulate_game_instance():

    num_die_rolls = 0
    # game_elements['go_increment'] = 100 # we should not be modifying this here. It is only for testing purposes.
    # One reason to modify go_increment is if your decision agent is not aggressively trying to monopolize. Since go_increment
    # by default is 200 it can lead to runaway cash increases for simple agents like ours.

    print('players will play in the following order: ','->'.join([p.player_name for p in env.game_elements['players']]))
    print('Beginning play. Rolling first die...')
    current_player_index = 0
    num_active_players = 4
    winner = None
    while num_active_players > 1:
        current_player = env.game_elements['players'][current_player_index]
        while current_player.status == 'lost':
            current_player_index += 1
            current_player_index = current_player_index % len(env.game_elements['players'])
            current_player = env.game_elements['players'][current_player_index]
        current_player.status = 'current_move'

        skip_turn = 0
        pre_roll_action = dict()
        pre_roll_action['player'] = current_player
        pre_roll_action['action'] = "make_pre_roll_moves"
        state, reward, done, info = env.step(pre_roll_action) ############
        if reward == 2:
            skip_turn += 1
        out_of_turn_player_index = current_player_index + 1
        out_of_turn_count = 0
        while skip_turn != num_active_players and out_of_turn_count<=200:
            out_of_turn_count += 1
            # print('checkpoint 1')
            out_of_turn_player = env.game_elements['players'][out_of_turn_player_index%len(env.game_elements['players'])]
            if out_of_turn_player.status == 'lost':
                out_of_turn_player_index += 1
                continue
            oot_action = dict()
            oot_action['player'] = out_of_turn_player
            oot_action['action'] = "make_out_of_turn_moves"
            state, reward, done, info = env.step(oot_action) ###############
            env.game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
            params = dict()
            params['self']=out_of_turn_player
            params['current_gameboard']=env.game_elements
            env.game_elements['history']['param'].append(params)
            env.game_elements['history']['return'].append(reward)

            if reward == 2:
                skip_turn += 1
            else:
                skip_turn = 0
            out_of_turn_player_index += 1

        # now we roll the dice and get into the post_roll phase,
        # but only if we're not in jail.

        r = env.roll_dice()
        env.game_elements['history']['function'].append(env.roll_dice)
        params = dict()
        params['die_objects'] = env.game_elements['dies']
        params['choice'] = np.random.choice
        env.game_elements['history']['param'].append(params)
        env.game_elements['history']['return'].append(r)

        num_die_rolls += 1
        env.game_elements['current_die_total'] = sum(r)
        print('dies have come up ',str(r))
        if not current_player.currently_in_jail:
            check_for_go = True
            env.move_player_after_dieroll(current_player, r, check_for_go)
            env.game_elements['history']['function'].append(env.move_player_after_dieroll)
            params = dict()
            params['player'] = current_player
            params['rel_move'] = sum(r)
            params['current_gameboard'] = env.game_elements
            params['check_for_go'] = check_for_go
            env.game_elements['history']['param'].append(params)
            env.game_elements['history']['return'].append(None)

            env.process_move_consequences_func(current_player)
            # add to game history
            env.game_elements['history']['function'].append(env.process_move_consequences_func)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = env.game_elements
            env.game_elements['history']['param'].append(params)
            env.game_elements['history']['return'].append(None)

            # post-roll for current player. No out-of-turn moves allowed at this point.
            post_roll_action = dict()
            post_roll_action['player'] = current_player
            post_roll_action['action'] = "make_post_roll_moves"
            state, reward, done, info = env.step(post_roll_action) ############
            # add to game history
            env.game_elements['history']['function'].append(post_roll_action)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = env.game_elements
            env.game_elements['history']['param'].append(params)
            env.game_elements['history']['return'].append(None)

        else:
            current_player.currently_in_jail = False # the player is only allowed to skip one turn (i.e. this one)

        if current_player.current_cash < 0:
            handle_neg_cash = dict()
            handle_neg_cash['player'] = current_player
            handle_neg_cash['action'] = "handle_negative_cash_balance"
            state, reward, done, info = env.step(handle_neg_cash) ############
            # add to game history
            env.game_elements['history']['function'].append(handle_neg_cash)
            params = dict()
            params['player'] = current_player
            params['current_gameboard'] = env.game_elements
            env.game_elements['history']['param'].append(params)
            env.game_elements['history']['return'].append(reward)
            if reward == -1 or current_player.current_cash < 0:
                current_player.begin_bankruptcy_proceedings(env.game_elements)
                # add to game history
                env.game_elements['history']['function'].append(current_player.begin_bankruptcy_proceedings)
                params = dict()
                params['self'] = current_player
                params['current_gameboard'] = env.game_elements
                env.game_elements['history']['param'].append(params)
                env.game_elements['history']['return'].append(None)

                num_active_players -= 1
                env.diagnostics_exec()  ##prints asset owners and cash balances of each player

                if num_active_players == 1:
                    for p in env.game_elements['players']:
                        if p.status != 'lost':
                            winner = p
                            p.status = 'won'

        else:
            current_player.status = 'waiting_for_move'

        current_player_index = (current_player_index+1)%len(env.game_elements['players'])

        if env.diagnostics_runaway_cash() > 300000:
            env.diagnostics_exec()  ##prints asset owners and cash balances of each player
            return

    # let's print some numbers
    print('number of dice rolls: ',str(num_die_rolls))
    print('printing final asset owners and final cash balances: ')
    env.diagnostics_exec()  ##prints asset owners and cash balances of each player

    if winner:
        print('We have a winner: ', winner.player_name)

    return


if __name__ == "__main__":

    player_decision_agents = dict()
    '''
    If you have a custom agent that you want one of your players to be, then assign "custom_agent" as the 
    value of relevant dictionary player key.
    eg: player_decision_agents['player_1'] = "custom_agent"
    '''

    #player_decision_agents['player_1'] = "custom_agent"
    player_decision_agents['player_1'] = "background_agent_v1"
    player_decision_agents['player_2'] = "background_agent_v1"
    player_decision_agents['player_3'] = "background_agent_v1"
    player_decision_agents['player_4'] = "simple_decision_agent_1"
    env = gym.make("monopoly-v0", player_decision_agents=player_decision_agents, game_schema='/media/shilpa/data/projects/gym-monopoly/examples/monopoly_game_schema_v1-2.json')
    for player in env.game_elements['players']:
        if player.agent == "will be set in the game file":
            print("Setting up custom agent for ", player.player_name)
            '''
            Set up your decision agent and assign it to the relevant player here.
            '''
            pass
        else:
            pass
    simulate_game_instance()
