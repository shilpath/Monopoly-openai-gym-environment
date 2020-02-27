

def will_property_complete_set(player, asset, current_gameboard):
    """

    :param player: Player instance
    :param asset: Location instance
    :return: Boolean. True if the asset will complete a color set for the player, False otherwise. For railroads
    (or utilities), returns true only if player owns all other railroads (or utilities)
    """
    if asset.color is None:
        if asset.loc_class == 'railroad':
            if player.num_railroads_possessed == 3:
                return True
        elif asset.loc_class == 'utility':
            if player.num_utilities_possessed == 1:
                return True
        else:
            print('This asset does not have a color and is neither utility nor railroad')
            raise Exception
    else:
        c = asset.color
        c_assets = current_gameboard['color_assets'][c]
        for c_asset in c_assets:
            if c_asset == asset:
                continue
            else:
                if c_asset not in player.assets:
                    return False
        return True # if we got here, then every asset of the color of 'asset' is possessed by player.


def identify_potential_mortgage(player, amount_to_raise, lone_constraint=False):
    """
    We return the property with the lowest mortgage such that it still exceeds or equals amount_to_raise, and if
    applicable, satisfies the lone constraint.
    :param player: Player instance. The potential mortgage has to be an unmortgaged property that this player owns.
    :param amount_to_raise: Integer. The amount of money looking to be raised from this mortgage.
    :param lone_constraint: Boolean. If true, we will limit our search to properties that meet the 'lone' constraint i.e.
    the property (if a railroad or utility) must be the only railroad or utility possessed by the player, or if colored,
    the property must be the only asset in its color class to be possessed by the player.
    :return: None, if a mortgage cannot be identified, otherwise a Location instance (representing the potential mortgage)
    """
    potentials = list()
    for a in player.assets:
        if a.is_mortgaged:
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        elif a.mortgage < amount_to_raise:
            continue
        elif lone_constraint:
            if is_property_lone(player, a):
                continue
        # a is a potential mortgage, and its mortgage price meets our fundraising bar.
        potentials.append((a,a.mortgage))

    if len(potentials) == 0:
        return None # nothing got identified
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1]) # sort by mortgage in ascending order
        return sorted_potentials[0][0]


def identify_potential_sale(player, amount_to_raise, lone_constraint=False):
    """
    All potential sales considered here will be to the bank. The logic is very similar to identify_potential_mortgage.
    We try to identify the cheapest property that will meet our fundraising bar (and if applicable, satisfy lone_constraint)
    :param player: Player instance. The potential sale has to be an unmortgaged property that this player owns.
    :param amount_to_raise: Integer. The amount of money looking to be raised from this sale.
    :param lone_constraint: Boolean. If true, we will limit our search to properties that meet the 'lone' constraint i.e.
    the property (if a railroad or utility) must be the only railroad or utility possessed by the player, or if colored,
    the property must be the only asset in its color class to be possessed by the player.
    :return: None, if a sale cannot be identified, otherwise a Location instance (representing the potential sale)
    """
    potentials = list()
    for a in player.assets:
        if a.is_mortgaged: # technically, we can sell a property even if it is mortgaged. If your agent wants to make
            # this distinction, you should modify this helper function. Note that cash received will be lower than
            # price/2 however, since you have to free the mortgage before you can sell.
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        elif a.price/2 < amount_to_raise:
            continue
        elif lone_constraint:
            if is_property_lone(player, a):
                continue
        # a is a potential sale, and its sale price meets our fundraising bar.
        potentials.append((a, a.price/2))

    if len(potentials) == 0:
        return None  # nothing got identified
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1])  # sort by sale price in ascending order
        return sorted_potentials[0][0]


def is_property_lone(player, asset):
    if asset.color is None:
        if asset.loc_class == 'railroad':
            if player.num_railroads_possessed == 1:
                return True
        elif asset.loc_class == 'utility':
            if player.num_utilities_possessed == 1:
                return True
        else:
            print('This asset does not have a color and is neither utility nor railroad')
            raise Exception
    else:
        c = asset.color
        for c_asset in player.assets:
            if c_asset == asset:
                continue
            else:
                if c_asset.loc_class == 'real_estate' and c_asset.color == c: # player has another property with this color
                    return False
        return True # if we got here, then only this asset (of its color class) is possessed by player.


def identify_improvement_opportunity(player, current_gameboard):
    """
    Identify an opportunity to improve a property by building a house or hotel. This is a 'strategic' function; there
    are many other ways/strategies to identify improvement opportunities than the one we use here.
    :param player:
    :param current_gameboard:
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.improve_property by the calling function.
    """
    potentials = list()
    for c in player.full_color_sets_possessed:
        c_assets = current_gameboard['color_assets'][c]
        for asset in c_assets:
            if can_asset_be_improved(asset,c_assets) and asset.price_per_house<=player.current_cash: # player must be able to afford the improvement
                potentials.append((asset,asset_incremental_improvement_rent(asset)-asset.price_per_house))
    if potentials:
        sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True) # sort in descending order
        param = dict()
        param['player'] = player
        param ['asset'] = sorted_potentials[0][0]
        param['current_gameboard'] = current_gameboard
        param['add_house'] = True
        param['add_hotel'] = False
        if param ['asset'].num_houses == 4:
            param['add_hotel'] = True
            param['add_house'] = False
        return param
    else:
        return None


def identify_sale_opportunity_to_player(player, current_gameboard):
    """
    Identify an opportunity to sell a property currently owned by player to another player by making a
    sell property offer. This is a 'strategic' function; there
    are many other ways/strategies to identify such sales than the one we use here. All we do is identify if
    there is a player who needs a single property to complete a full color set and if that property is a 'lone'
    property for us. If such a player exists for some such
    property that we own, we offer it to the player at 50% markup. We do not offer mortgaged properties for sale.
    For simplicity, we do not offer railroads or utilities for sale either. Other agents may consider more sophisticated
    strategies to handle railroads and utilities.
    :param player:
    :param current_gameboard:
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function.
    """
    potentials = list()
    for a in player.assets:
        if a.loc_class != 'real_estate' or a.is_mortgaged:
            continue
        if a.color in player.full_color_sets_possessed:
            continue
        if is_property_lone(player, a):
            for p in current_gameboard['players']:
                if p == player or p.status == 'lost':
                    continue
                elif will_property_complete_set(p, a, current_gameboard):
                    # we make an offer!
                    param = dict()
                    param['from_player'] = player
                    param['asset'] = a
                    param['to_player'] = p
                    param['price'] = a.price*1.5 # 50% markup on market price.
                    if param['price'] < param['to_player'].current_cash / 2:
                        param['price'] = param['to_player'].current_cash / 2  # how far would you go for a monopoly?
                    elif param['price'] > param['to_player'].current_cash:
                        # no point offering this to the player; they don't have money.
                        continue
                    potentials.append((param, param['price']))

    if not potentials:
        return None
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True)  # sort in descending order
        return sorted_potentials[0][0]


def can_asset_be_improved(asset, same_color_assets):
    """
    This function does not check if all the same colored assets are owned by the same player. This is something that
    should have been checked much earlier in the code. All that we check here is whether it is permissible to improve
    asset under the assumption that the asset, and all other assets of that color, belong to one player. We also do
    not check here whether the game board is in an incorrect state (i.e. if somehow the uniform development rule
    has been violated).

    We are also not checking affordability of the improvement since the player is not specified.
    :param asset:
    :param same_color_assets:
    :return:
    """
    if asset.loc_class != 'real_estate' or asset.is_mortgaged:
        return False
    if asset.num_hotels > 0:
        return False # we can't improve any further
    if asset.num_houses == 0:
        return True
    count = 0
    for c_asset in same_color_assets:
        if c_asset.color != asset.color:
            print('asset color is not the same as the color of the set')
            raise Exception # if this has happened, it probably indicates a problem in the code. That's why we don't return false
        if c_asset == asset:
            continue
        if c_asset.num_hotels > 0 and asset.num_houses == 4 :
            return True # we can build a hotel on asset
        if c_asset.num_houses > asset.num_houses:
            return True
        if c_asset.num_houses == asset.num_houses:
            count += 1

    if count == len(same_color_assets) - 1: # every asset in same_color has the same no. of houses as the current asset, hence
        # it can be improved (either by building another house, or a hotel).
        return True

    return False


def asset_incremental_improvement_rent(asset):
    """
    If we were to incrementally improve this asset, how much extra rent would we get?
    :param asset: the property to be (hypothetically) incrementally improved
    :return: Integer representing the additional rent we get if we were to incrementally improve this property. Note that
    we do not check if we 'can' improve it, we return assuming that we can.
    """
    if asset.num_hotels > 0:
        raise Exception # there is no incremental improvement possible. how did we get here?
    if asset.num_houses == 4:
        return asset.rent_hotel-asset.rent_4_houses
    elif asset.num_houses == 3:
        return asset.rent_4_houses - asset.rent_3_houses
    elif asset.num_houses == 2:
        return asset.rent_3_houses - asset.rent_2_houses
    elif asset.num_houses == 1:
        return asset.rent_2_houses - asset.rent_1_house
    else:
        return asset.rent_1_house - (asset.rent*2) # remember, if the house can be improved, then it is monopolized, so twice the rent is being charged even without houses.







