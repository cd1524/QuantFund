import os
from py_clob_client.constants import POLYGON
from py_clob_client.client import ClobClient

import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, market_name, query_status, action):

    #data is a list of lists, where each list is a pair of timestamp and price
    data = np.array(data)
    timestamps = data[:,0]
    prices = data[:,1]
    plt.plot(timestamps, prices)
    plt.title(f"{market_name} - {action}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig(f"figures/{market_name}_{action}.png")

    return

def get_markets_by_keyword(client, keyword):
    data = client.get_sampling_markets()
    markets = data['data']
    next_cursor = data['next_cursor'] # extend the search for the next set of results
    possible_markets = []
    for market in markets:
        # this bottom bit is find the market by query
        question = market['question']
        split_question = question.split(' ')
        #convert all to lower case
        split_question = [item.lower() for item in split_question]
        if (keyword.lower() in split_question) and (not market['closed']) and (market['active']):
            possible_markets.append(market)
    return possible_markets

if __name__ =="__main__":

    host = "https://clob.polymarket.com"
    key = '0x1bd6c7ae06269db6bf0f8b4412a029637158aa7b93e7501a175faa5d6316315d'
    chain_id = POLYGON

    # Create CLOB client and get/set API credentials
    client = ClobClient(host, key=key, chain_id=chain_id)
    client.set_api_creds(client.create_or_derive_api_creds())

    keyword = 'trump'
    query_status = False

    #possible time frames 
    # 1m: 1 month
    # 1w: 1 week
    # 1d: 1 day
    # 6h: 6 hours
    # 1h: 1 hour
    # max: max range
    time_frame = 'max'
    resolution = '60' # 60 min

    # Get market by keyword=
    filtered_markets = get_markets_by_keyword(client, keyword)

    #we need to plot the price history of the market

    for market in filtered_markets:
        tokens = market['tokens']
        for token in tokens:
            # now we have the token, we need to get the price history
            historic_data = client.get_price_history_with_interval(token['token_id'], time_frame, resolution)['history']
            # we need to plot the data
            historic_data_list = [[item['t'],item['p']] for item in historic_data]
            plot_data(historic_data_list, market['question'], query_status=None, action=token['outcome'])
            plt.clf()




    