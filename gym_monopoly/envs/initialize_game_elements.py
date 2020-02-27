"""
The only public facing function is initialize_board. All _initialize_* functions are only for internal use. If you
want to play around, you could always implement your _initialize functions and replace accordingly in initialize_board!
"""

from gym_monopoly.envs import location
from gym_monopoly.envs.dice import Dice
from gym_monopoly.envs.bank import Bank
from gym_monopoly.envs.card_utility_actions import * # functions from this module will be used in reflections in initialize_board,
                                    # and excluding this import will lead to run-time errors
import sys
from gym_monopoly.envs.player import Player
from gym_monopoly.envs import card


def initialize_board(game_schema, player_decision_agents):

    game_elements = dict()
    print('Beginning game set up...')

    # Step 0: initialize bank
    _initialize_bank(game_elements)
    print('Successfully instantiated and initialized bank.')

    # Step 1: set locations
    _initialize_locations(game_elements, game_schema)
    print('Successfully instantiated and initialized all locations on board.')


    # Step 2: set dice
    _initialize_dies(game_elements, game_schema)
    print('Successfully instantiated and initialized dies')

    # Step 3: set cards
    _initialize_cards(game_elements, game_schema)
    print('Successfully instantiated and initialized cards')

    # Step 4: set players
    _initialize_players(game_elements, game_schema, player_decision_agents)
    print('Successfully instantiated and initialized players and decision agents')

    # Step 5: set history data structures
    _initialize_game_history_structs(game_elements)
    print('Successfully instantiated game history data structures')

    return game_elements


def _initialize_bank(game_elements):
    game_elements['bank'] = Bank()


def _initialize_locations(game_elements, game_schema):
    location_objects = dict() # key is a location name, and value is a Location object
    railroad_positions = list() # list of integers, with each integer corresponding to a railroad location in
    # game_elements['location_sequence']
    utility_positions = list() # list of integers, with each integer corresponding to a utility location in
    # game_elements['location_sequence']
    location_sequence = list() # list of Location objects in sequence, as they would be ordered on a linear game board.
    color_assets = dict()  # key is a string color (of a real estate property) and value is the set of location objects
    # that have that color. Any asset that does not have a color or where the color is None in the schema will not be
    # included in any set, since we do not insert None in as a key

    for l in game_schema['locations']['location_states']:

        if l['loc_class'] == 'action':
            action_args = l.copy()
            action_args['perform_action'] = getattr(sys.modules[__name__], l['perform_action'])
            location_objects[l['name']] = location.ActionLocation(**action_args)

        elif l['loc_class'] == 'do_nothing':
            location_objects[l['name']] = location.DoNothingLocation(**l)

        elif l['loc_class'] == 'real_estate':
            real_estate_args = l.copy()
            real_estate_args['owned_by'] = game_elements['bank']
            real_estate_args['num_houses'] = 0
            real_estate_args['num_hotels'] = 0
            location_objects[l['name']] = location.RealEstateLocation(**real_estate_args)

        elif l['loc_class'] == 'tax':
            location_objects[l['name']] = location.TaxLocation(**l)

        elif l['loc_class'] == 'railroad':
            railroad_args = l.copy()
            railroad_args['owned_by'] = game_elements['bank']
            location_objects[l['name']] = location.RailroadLocation(**railroad_args)

        elif l['loc_class'] == 'utility':
            utility_args = l.copy()
            utility_args['owned_by'] = game_elements['bank']
            location_objects[l['name']] = location.UtilityLocation(**utility_args)

        else:
            print('encountered unexpected location class: ', l['loc_class'])
            raise Exception

    for i in range(0, len(game_schema['location_sequence'])):
        loc = location_objects[game_schema['location_sequence'][i]]
        location_sequence.append(loc) # we first get the name of
        # the location at index i of the game schema, and then use it in location_objects to get the actual location
        # object (loc) corresponding to that location name. We then append it to location_sequence. The net result is
        # that we have gone from a sequence of location names to the corresponding sequence of objects.
        if loc.loc_class == 'railroad':
            railroad_positions.append(i)
        elif loc.loc_class == 'utility':
            utility_positions.append(i)
        elif loc.name == 'In Jail/Just Visiting':
            game_elements['jail_position'] = i

    game_elements['railroad_positions'] = railroad_positions
    game_elements['utility_positions'] = utility_positions

    if len(location_sequence) != game_schema['locations']['location_count']:
        print('location count: ', str(game_schema['locations']['location_count']), ', length of location sequence: ',
        str(len(location_sequence)), ' are unequal.')
        raise Exception

    if location_sequence[game_schema['go_position']].name != 'Go':
        print('go positions are not aligned')
        raise Exception
    else:
        game_elements['go_position'] = game_schema['go_position']
        game_elements['go_increment'] = game_schema['go_increment']

    game_elements['location_objects'] = location_objects
    game_elements['location_sequence'] = location_sequence

    for o in location_sequence:
        if o.color is None:
            continue
        elif o.color not in game_schema['players']['player_states']['full_color_sets_possessed'][0]:
            print(o.color)
            raise Exception
        else:
            if o.color not in color_assets:
                color_assets[o.color] = set()
            color_assets[o.color].add(o)

    game_elements['color_assets'] = color_assets


def _initialize_dies(game_elements, game_schema):
    if len(game_schema['die']['die_state']) != game_schema['die']['die_count']:
        print('dice count is unequal to number of specified dice state-vectors...')
        raise Exception
    die_count = game_schema['die']['die_count']
    die_objects = list()
    for i in range(0, die_count):
        die_objects.append(Dice(game_schema['die']['die_state'][i])) # send in the vector

    game_elements['dies'] = die_objects
    game_elements['current_die_total'] = 0


def _initialize_cards(game_elements, game_schema):
    community_chest_cards = set() # community chest card objects
    chance_cards = set() # chance card objects

    community_chest_card_objects = dict() # key is a community chest card name and value is an object
    chance_card_objects = dict() # key is a chance card name and value is an object

    # note that the number of keys in community_chest_card_objects may be different from the number of items in
    # community_chest_cards (same for chance), since if there is more than one card with the same name, we will end up having
    # fewer keys in the _objects data structure. In the _cards data structure we directly add the objects, so even
    # if two cards share a name, they will be treated as distinct objects. We do an additional check after the for
    # loop to verify that we have the expected numbers of cards as specified in the game schema.

    for specific_card in game_schema['cards']['community_chest']['card_states']:
        card_obj = None
        if specific_card['card_type'] == 'movement':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_args['destination'] = game_elements['location_objects'][specific_card['destination']]
                card_obj = card.MovementCard(**card_args)
                community_chest_cards.add(card_obj)

        elif specific_card['card_type'] == 'contingent_movement':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.ContingentMovementCard(**card_args)
                community_chest_cards.add(card_obj)

        elif specific_card['card_type'] == 'positive_cash_from_bank' or specific_card[
            'card_type'] == 'negative_cash_from_bank':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.CashFromBankCard(**card_args)
                community_chest_cards.add(card_obj)

        elif specific_card['card_type'] == 'contingent_cash_from_bank':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_args['contingency'] = getattr(sys.modules[__name__], specific_card['contingency'])
                card_obj = card.ContingentCashFromBankCard(**card_args)
                community_chest_cards.add(card_obj)

        elif specific_card['card_type'] == 'positive_cash_from_players' or card[
            'card_type'] == 'negative_cash_from_players':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.CashFromPlayersCard(**card_args)
                community_chest_cards.add(card_obj)
        else:

            print('community chest card type is not recognized: ', specific_card['card_type'])
            raise Exception

        community_chest_card_objects[card_obj.name] = card_obj

    if len(community_chest_cards) != game_schema['cards']['community_chest']['card_count']:
        print('community chest card count and number of items in community chest card set are inconsistent')

    for specific_card in game_schema['cards']['chance']['card_states']:
        card_obj = None
        if specific_card['card_type'] == 'movement':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_args['destination'] = game_elements['location_objects'][specific_card['destination']]
                card_obj = card.MovementCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'movement_payment':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.MovementPaymentCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'contingent_movement':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.ContingentMovementCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'movement_relative':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.MovementRelativeCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'positive_cash_from_bank' or specific_card[
            'card_type'] == 'negative_cash_from_bank':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.CashFromBankCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'contingent_cash_from_bank':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_args['contingency'] = getattr(sys.modules[__name__], specific_card['contingency'])
                card_obj = card.ContingentCashFromBankCard(**card_args)
                chance_cards.add(card_obj)

        elif specific_card['card_type'] == 'positive_cash_from_players' or specific_card[
            'card_type'] == 'negative_cash_from_players':
            for i in range(0, specific_card['num']):
                card_args = specific_card.copy()
                del card_args['num']
                card_args['action'] = getattr(sys.modules[__name__], specific_card['action'])
                card_obj = card.CashFromPlayersCard(**card_args)
                chance_cards.add(card_obj)
        else:
            print('chance card type is not recognized: ', specific_card['card_type'])
            raise Exception

        chance_card_objects[card_obj.name] = card_obj

    if len(chance_cards) != game_schema['cards']['chance']['card_count']:
        print('chance card count and number of items in chance card set are inconsistent')

    game_elements['chance_cards'] = chance_cards
    game_elements['community_chest_cards'] = community_chest_cards
    game_elements['chance_card_objects'] = chance_card_objects
    game_elements['community_chest_card_objects'] = community_chest_card_objects


def _initialize_players(game_elements, game_schema, player_decision_agents):
    players = list()
    player_dict = game_schema['players']['player_states']
    for player in player_dict['player_name']:
        player_args = dict()
        player_args['status'] = 'waiting_for_move'
        player_args['current_position'] = game_schema['go_position']
        player_args['has_get_out_of_jail_chance_card'] = False
        player_args['has_get_out_of_jail_community_chest_card'] = False
        player_args['current_cash'] = player_dict['starting_cash']
        player_args['num_railroads_possessed'] = 0
        player_args['num_utilities_possessed'] = 0
        player_args['full_color_sets_possessed'] = set()
        player_args['assets'] = set()
        player_args['currently_in_jail'] = False

        player_args['player_name'] = player
        if player_decision_agents[player] != "custom_agent":
            player_args['agent'] = player_decision_agents[player]
        else:
            player_args['agent'] = "will be set in the game file"
            print("Initialize your agent with your respective logic")

        players.append(Player(**player_args))

    game_elements['players'] = players


def _initialize_game_history_structs(game_elements):
    game_elements['history'] = dict()
    game_elements['history']['function'] = list()
    game_elements['history']['param'] = list()
    game_elements['history']['return'] = list()
