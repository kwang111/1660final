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
from datetime import datetime

"""
* ECON1660
* Final Project: Pandemmy Foodies :3
"""

# Expand this:
food_business_list0 = ['restaurant', 'food', 'cafe', 'pub']
food_business_list = ['restaurant', 'food', 'cafe', 'pub', 'Afghan', 'African', 'Senegalese', 'South African', 'American (New)', 'American (Traditional)', 'Arabian', 'Argentine', 'Armenian', 'Asian Fusion', 'Australian', 'Austrian', 'Bangladeshi', 'Barbeque', 'Basque', 'Belgian', 'Brasseries', 'Brazilian', 'Breakfast & Brunch', 'Pancakes', 'British', 'Buffets', 'Bulgarian', 'Burgers', 'Burmese', 'Cafes', 'Themed Cafes', 'Cafeteria', 'Cajun/Creole', 'Cambodian', 'Caribbean', 'Dominican', 'Haitian', 'Puerto Rican', 'Trinidadian', 'Catalan', 'Cheesesteaks', 'Chicken Shop', 'Chicken Wings', 'Chinese', 'Cantonese', 'Dim Sum', 'Hainan', 'Shanghainese', 'Szechuan', 'Comfort Food', 'Creperies', 'Cuban', 'Czech', 'Delis', 'Diners', 'Dinner Theater', 'Eritrean', 'Ethiopian', 'Fast Food', 'Filipino', 'Fish & Chips', 'Fondue', 'Food Court', 'Food Stands', 'French', 'Mauritius', 'Reunion', 'Game Meat', 'Gastropubs', 'Georgian', 'German', 'Gluten-Free', 'Greek', 'Guamanian', 'Halal', 'Hawaiian', 'Himalayan/Nepalese', 'Honduran', 'Hong Kong Style Cafe', 'Hot Dogs',
                      'Hot Pot', 'Hungarian', 'Iberian', 'Indian', 'Indonesian', 'Irish', 'Italian', 'Calabrian', 'Sardinian', 'Sicilian', 'Tuscan', 'Japanese', 'Conveyor Belt Sushi', 'Izakaya', 'Japanese Curry', 'Ramen', 'Teppanyaki', 'Kebab', 'Korean', 'Kosher', 'Laotian', 'Latin American', 'Colombian', 'Salvadoran', 'Venezuelan', 'Live/Raw Food', 'Malaysian', 'Mediterranean', 'Falafel', 'Mexican', 'Tacos', 'Middle Eastern', 'Egyptian', 'Lebanese', 'Modern European', 'Mongolian', 'Moroccan', 'New Mexican Cuisine', 'Nicaraguan', 'Noodles', 'Pakistani', 'Pan Asia', 'Persian/Iranian', 'Peruvian', 'Pizza', 'Polish', 'Polynesian', 'Pop-Up Restaurants', 'Portuguese', 'Poutineries', 'Russian', 'Salad', 'Sandwiches', 'Scandinavian', 'Scottish', 'Seafood', 'Singaporean', 'Slovakian', 'Somali', 'Soul Food', 'Soup', 'Southern', 'Spanish', 'Sri Lankan', 'Steakhouses', 'Supper Clubs', 'Sushi Bars', 'Syrian', 'Taiwanese', 'Tapas Bars', 'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Turkish', 'Ukrainian', 'Uzbek', 'Vegan', 'Vegetarian', 'Vietnamese', 'Waffles', 'Wraps']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekends = ['Saturday', 'Sunday']
days_of_week = weekdays + weekends
major_cities = ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Washington', 'Dallas', 'Houston', 'Boston', 'Philadelphia', 'Atlanta', 'Seattle']
states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rbd", "--readbusinessdata",
                        help="Read Business Data", action="store_true")
    parser.add_argument("-rrd", "--readreviewdata",
                        help="Read Review Data", action="store_true")
    return parser.parse_args()


def load_business_data(path):
    business_list_lasso = []
    business_list_data = []
    attributes_list = []
    print("Reading in Business JSON File")
    with open(path) as f:
        for business in f:
            business_dict = json.loads(business)
            # Check if business is restaurant
            categories = business_dict["categories"]
            if categories != None and any(x in categories for x in food_business_list):
                business_id = business_dict["business_id"]
                business_features = {}
                # Clean up features of the business
                # Making indicators
                for city in major_cities:
                    if business_dict["city"] == city:
                        business_features.update({city: 1})
                    else:
                        business_features.update({city: 0})
                for state in states:
                    if business_dict["state"] == state:
                        business_features.update({state: 1})
                    else:
                        business_features.update({state: 0})
                business_features.update({"stars": business_dict["stars"]})
                business_features.update(
                    {"review_count": business_dict["review_count"]})
                business_features.update({"is_open": business_dict["is_open"]})
                # Do stuff to attributes to create features
                if business_dict["attributes"] != None:
                    for attribute in business_dict["attributes"].keys():
                        attributes_list.append(attribute) if attribute not in attributes_list else attributes_list
                # Do stuff to hours to create features
                open_days = 0
                for day in days_of_week:
                    if (business_dict["hours"] != None) and (day in business_dict["hours"].keys()):
                        d1, d2 = business_dict["hours"][day].split('-')
                        d1 = datetime.strptime(d1, '%H:%M')
                        d2 = datetime.strptime(d2, '%H:%M')
                        business_features.update({day:int((d2-d1).seconds / 60)})
                        open_days += 1
                    else:
                        business_features.update({day:0})
                # Summing up hours
                total = 0
                for day in days_of_week:
                    total += business_features[day]
                business_features.update({"total_hours":total})
                # check weekday/weekend
                wd = 0
                we = 0
                for weekday in weekdays:
                    if business_features[weekday] > 0:
                        wd = 1
                for weekend in weekends:
                    if business_features[weekend] > 0:
                        we = 1
                business_features.update({"weekday":wd})
                business_features.update({"weekend":we})
                business_features.update({"num_open_days":open_days})

                business_list_lasso.append((business_id, business_features))
                business_list_data.append((business_id, business_dict))
    # for i in range(10):
    #     print(business_list[i]["name"])
        # print(business_list[i]["categories"])
    print(business_list_lasso[500][1]["Wednesday"])
    print(attributes_list)
    # print(len(business_list))
    f.close()
    return business_list_lasso, business_list_data


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
