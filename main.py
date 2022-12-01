import csv
from math import log
from collections import defaultdict, Counter
import this
from datetime import datetime
import statistics
import string
import random
import pickle
import json
import argparse

"""
* ECON1660
* Final Project: Pandemmy Foodies :3
"""

# Expand this:
food_business_list0 = ['restaurant', 'food', 'cafe', 'pub']
food_business_list = ['restaurant', 'food', 'cafe', 'pub', 'Afghan', 'African', 'Senegalese', 'South African', 'American (New)', 'American (Traditional)', 'Arabian', 'Argentine', 'Armenian', 'Asian Fusion', 'Australian', 'Austrian', 'Bangladeshi', 'Barbeque', 'Basque', 'Belgian', 'Brasseries', 'Brazilian', 'Breakfast & Brunch', 'Pancakes', 'British', 'Buffets', 'Bulgarian', 'Burgers', 'Burmese', 'Cafes', 'Themed Cafes', 'Cafeteria', 'Cajun/Creole', 'Cambodian', 'Caribbean', 'Dominican', 'Haitian', 'Puerto Rican', 'Trinidadian', 'Catalan', 'Cheesesteaks', 'Chicken Shop', 'Chicken Wings', 'Chinese', 'Cantonese', 'Dim Sum', 'Hainan', 'Shanghainese', 'Szechuan', 'Comfort Food', 'Creperies', 'Cuban', 'Czech', 'Delis', 'Diners', 'Dinner Theater', 'Eritrean', 'Ethiopian', 'Fast Food', 'Filipino', 'Fish & Chips', 'Fondue', 'Food Court', 'Food Stands', 'French', 'Mauritius', 'Reunion', 'Game Meat', 'Gastropubs', 'Georgian', 'German', 'Gluten-Free', 'Greek', 'Guamanian', 'Halal', 'Hawaiian', 'Himalayan/Nepalese', 'Honduran', 'Hong Kong Style Cafe', 'Hot Dogs',
                      'Hot Pot', 'Hungarian', 'Iberian', 'Indian', 'Indonesian', 'Irish', 'Italian', 'Calabrian', 'Sardinian', 'Sicilian', 'Tuscan', 'Japanese', 'Conveyor Belt Sushi', 'Izakaya', 'Japanese Curry', 'Ramen', 'Teppanyaki', 'Kebab', 'Korean', 'Kosher', 'Laotian', 'Latin American', 'Colombian', 'Salvadoran', 'Venezuelan', 'Live/Raw Food', 'Malaysian', 'Mediterranean', 'Falafel', 'Mexican', 'Tacos', 'Middle Eastern', 'Egyptian', 'Lebanese', 'Modern European', 'Mongolian', 'Moroccan', 'New Mexican Cuisine', 'Nicaraguan', 'Noodles', 'Pakistani', 'Pan Asia', 'Persian/Iranian', 'Peruvian', 'Pizza', 'Polish', 'Polynesian', 'Pop-Up Restaurants', 'Portuguese', 'Poutineries', 'Russian', 'Salad', 'Sandwiches', 'Scandinavian', 'Scottish', 'Seafood', 'Singaporean', 'Slovakian', 'Somali', 'Soul Food', 'Soup', 'Southern', 'Spanish', 'Sri Lankan', 'Steakhouses', 'Supper Clubs', 'Sushi Bars', 'Syrian', 'Taiwanese', 'Tapas Bars', 'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Turkish', 'Ukrainian', 'Uzbek', 'Vegan', 'Vegetarian', 'Vietnamese', 'Waffles', 'Wraps']


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rbd", "--readbusinessdata",
                        help="Read Business Data", action="store_true")
    parser.add_argument("-rrd", "--readreviewdata",
                        help="Read Review Data", action="store_true")
    return parser.parse_args()


def load_business_data(path):
    business_list = []
    print("Reading in Business JSON File")
    with open(path) as f:
        for business in f:
            business_dict = json.loads(business)
            # Check if business is restaurant
            categories = business_dict["categories"]
            if categories != None and any(x in categories for x in food_business_list):
                business_dict_new = {}
                # Clean up features of the business
                business_dict_new.update({"id": business_dict["business_id"]})
                business_dict_new.update({"name": business_dict["name"]})
                business_dict_new.update({"city": business_dict["city"]})
                business_dict_new.update({"state": business_dict["state"]})
                business_dict_new.update({"po": business_dict["postal_code"]})
                business_dict_new.update({"stars": business_dict["stars"]})
                business_dict_new.update(
                    {"review_count": business_dict["review_count"]})
                business_dict_new.update({"is_open": business_dict["is_open"]})
                # Do stuff to attributes to create features
                # Do sttuff to hours to create features
                business_list.append(business_dict_new)
    # for i in range(10):
    #     print(business_list[i]["name"])
        # print(business_list[i]["categories"])
    print(len(business_list))
    f.close()
    return business_list


def main(args):
    if args.readbusinessdata:
        business_list = load_business_data(
            "yelp_dataset/yelp_academic_dataset_business.json")
        with open('business_list', 'wb') as file:
            pickle.dump(business_list, file)
    else:
        # If not reading in / processing data again, load the data
        with open('loans_AB_processed', 'rb') as file:
            business_list = pickle.load(file)


if __name__ == '__main__':
    args = parseArguments()
    main(args)
