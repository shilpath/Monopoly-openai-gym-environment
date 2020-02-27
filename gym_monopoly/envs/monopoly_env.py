import gym
from gym import error, spaces, utils
from gym.utils import seeding
import json
import numpy as np
from gym_monopoly.envs import initialize_game_elements
from gym_monopoly.envs.action_choices import roll_die
from gym_monopoly.envs.card_utility_actions import move_player_after_die_roll
from gym_monopoly.envs import simple_decision_agent_1
from gym_monopoly.envs import diagnostics
from gym_monopoly.envs.agent import Agent
from gym_monopoly.envs import background_agent_v1

class MonopolyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, player_decision_agents, game_schema):
        self.player_decision_agents = dict()

        if player_decision_agents['player_1'] == "background_agent_v1":
            self.player_decision_agents['player_1'] = Agent(**background_agent_v1.decision_agent_methods)
        elif player_decision_agents['player_1'] == "simple_decision_agent_1":
            self.player_decision_agents['player_1'] = Agent(**simple_decision_agent_1.decision_agent_methods)
        elif player_decision_agents['player_1'] == "custom_agent":
            self.player_decision_agents['player_1'] = "custom_agent"

        if player_decision_agents['player_2'] == "background_agent_v1":
            self.player_decision_agents['player_2'] = Agent(**background_agent_v1.decision_agent_methods)
        elif player_decision_agents['player_2'] == "simple_decision_agent_1":
            self.player_decision_agents['player_2'] = Agent(**simple_decision_agent_1.decision_agent_methods)
        elif player_decision_agents['player_2'] == "custom_agent":
            self.player_decision_agents['player_2'] = "custom_agent"

        if player_decision_agents['player_3'] == "background_agent_v1":
            self.player_decision_agents['player_3'] = Agent(**background_agent_v1.decision_agent_methods)
        elif player_decision_agents['player_3'] == "simple_decision_agent_1":
            self.player_decision_agents['player_3'] = Agent(**simple_decision_agent_1.decision_agent_methods)
        elif player_decision_agents['player_3'] == "custom_agent":
            self.player_decision_agents['player_3'] = "custom_agent"

        if player_decision_agents['player_4'] == "background_agent_v1":
            self.player_decision_agents['player_4'] = Agent(**background_agent_v1.decision_agent_methods)
        elif player_decision_agents['player_4'] == "simple_decision_agent_1":
            self.player_decision_agents['player_4'] = Agent(**simple_decision_agent_1.decision_agent_methods)
        elif player_decision_agents['player_4'] == "custom_agent":
            self.player_decision_agents['player_4'] = "custom_agent"

        self.game_schema_path = game_schema
        self.game_elements = MonopolyEnv.set_up_board(self.game_schema_path, self.player_decision_agents)
        print("BOARD is SET UP")
        self.seed(4)
        self.reset()
        #self.action_space =
        #self.observation_space =


    def seed(self, np_seed = None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        np.random.seed(np_seed)
        np.random.shuffle(self.game_elements['players'])
        self.game_elements['seed'] = np_seed
        self.game_elements['card_seed'] = np_seed
        self.game_elements['choice_function'] = np.random.choice
        return #should have returned the seeds, but here the seeds are updated as part of game_elements, hence not returning the seeds explicitly

    def step(self, action_dict):
        reward = 0
        if action_dict['action'] == "make_pre_roll_moves":
            reward = action_dict['player'].make_pre_roll_moves(self.game_elements)
        elif action_dict['action'] == "make_out_of_turn_moves":
            reward = action_dict['player'].make_out_of_turn_moves(self.game_elements)
        elif action_dict['action'] == "make_post_roll_moves":
            reward = action_dict['player'].make_post_roll_moves(self.game_elements)
        elif action_dict['action'] == "handle_negative_cash_balance":
            reward = action_dict['player'].agent.handle_negative_cash_balance(action_dict['player'], self.game_elements)

        state = self.game_elements
        done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        pass

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    @staticmethod
    def set_up_board(game_schema_file_path, player_decision_agents):
        game_schema = json.load(open(game_schema_file_path, 'r'))
        return initialize_game_elements.initialize_board(game_schema, player_decision_agents)


    def roll_dice(self):
        r = roll_die(self.game_elements['dies'], np.random.choice)
        return r

    def move_player_after_dieroll(self, current_player, r, check_for_go):
        move_player_after_die_roll(current_player, sum(r), self.game_elements, check_for_go)

    def process_move_consequences_func(self, current_player):
        current_player.process_move_consequences(self.game_elements)

    def diagnostics_exec(self):
        diagnostics.print_asset_owners(self.game_elements)
        diagnostics.print_player_cash_balances(self.game_elements)

    def diagnostics_runaway_cash(self):
        return diagnostics.max_cash_balance(self.game_elements)
