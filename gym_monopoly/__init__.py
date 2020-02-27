from gym.envs.registration import register

register(
    id='monopoly-v0',
    entry_point='gym_monopoly.envs:MonopolyEnv',
    kwargs = {'player_decision_agents': {}, 'game_schema': "/media/shilpa/data/projects/gym-monopoly/examples/monopoly_game_schema_v1-2.json"},
)

