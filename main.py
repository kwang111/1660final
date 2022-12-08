import csv
from math import log
from collections import defaultdict, Counter
from tabnanny import check
import this
from datetime import datetime
import statistics
import string
import random
import pickle
import json
import argparse
from datetime import datetime
import ast

"""
* ECON1660
* Final Project: Pandemmy Foodies :3
"""

# Expand this:
food_business_list = ['restaurant', 'food', 'cafe', 'pub', 'Afghan', 'African', 'Senegalese', 'South African', 'American (New)', 'American (Traditional)', 'Arabian', 'Argentine', 'Armenian', 'Asian Fusion', 'Australian', 'Austrian', 'Bangladeshi', 'Barbeque', 'Basque', 'Belgian', 'Brasseries', 'Brazilian', 'Breakfast & Brunch', 'Pancakes', 'British', 'Buffets', 'Bulgarian', 'Burgers', 'Burmese', 'Cafes', 'Themed Cafes', 'Cafeteria', 'Cajun/Creole', 'Cambodian', 'Caribbean', 'Dominican', 'Haitian', 'Puerto Rican', 'Trinidadian', 'Catalan', 'Cheesesteaks', 'Chicken Shop', 'Chicken Wings', 'Chinese', 'Cantonese', 'Dim Sum', 'Hainan', 'Shanghainese', 'Szechuan', 'Comfort Food', 'Creperies', 'Cuban', 'Czech', 'Delis', 'Diners', 'Dinner Theater', 'Eritrean', 'Ethiopian', 'Fast Food', 'Filipino', 'Fish & Chips', 'Fondue', 'Food Court', 'Food Stands', 'French', 'Mauritius', 'Reunion', 'Game Meat', 'Gastropubs', 'Georgian', 'German', 'Gluten-Free', 'Greek', 'Guamanian', 'Halal', 'Hawaiian', 'Himalayan/Nepalese', 'Honduran', 'Hong Kong Style Cafe', 'Hot Dogs',
                      'Hot Pot', 'Hungarian', 'Iberian', 'Indian', 'Indonesian', 'Irish', 'Italian', 'Calabrian', 'Sardinian', 'Sicilian', 'Tuscan', 'Japanese', 'Conveyor Belt Sushi', 'Izakaya', 'Japanese Curry', 'Ramen', 'Teppanyaki', 'Kebab', 'Korean', 'Kosher', 'Laotian', 'Latin American', 'Colombian', 'Salvadoran', 'Venezuelan', 'Live/Raw Food', 'Malaysian', 'Mediterranean', 'Falafel', 'Mexican', 'Tacos', 'Middle Eastern', 'Egyptian', 'Lebanese', 'Modern European', 'Mongolian', 'Moroccan', 'New Mexican Cuisine', 'Nicaraguan', 'Noodles', 'Pakistani', 'Pan Asia', 'Persian/Iranian', 'Peruvian', 'Pizza', 'Polish', 'Polynesian', 'Pop-Up Restaurants', 'Portuguese', 'Poutineries', 'Russian', 'Salad', 'Sandwiches', 'Scandinavian', 'Scottish', 'Seafood', 'Singaporean', 'Slovakian', 'Somali', 'Soul Food', 'Soup', 'Southern', 'Spanish', 'Sri Lankan', 'Steakhouses', 'Supper Clubs', 'Sushi Bars', 'Syrian', 'Taiwanese', 'Tapas Bars', 'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Turkish', 'Ukrainian', 'Uzbek', 'Vegan', 'Vegetarian', 'Vietnamese', 'Waffles', 'Wraps']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekends = ['Saturday', 'Sunday']
days_of_week = weekdays + weekends
major_cities = ['New York', 'Los Angeles', 'Chicago', 'San Francisco',
                'Washington', 'Dallas', 'Houston', 'Boston', 'Philadelphia', 'Atlanta', 'Seattle']
states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
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


def load_checkin_data(path):
    checkin_dict = {}

    with open(path) as f:
        for b in f:
            business = json.loads(b)
            business_id = business["business_id"].replace('-', '')
            checkin_dates = business["date"].split(',')
            last_checkin = checkin_dates[-1].replace(' ', '')[0:4]
            checkin_dict.update({business_id: int(last_checkin)})
    return checkin_dict


def load_business_data(path, checkin_dict):
    business_list_lasso = []
    business_list_data = []
    print("Reading in Business JSON File")
    business_counter = 0
    closed_business_counter = 0
    with open(path) as f:
        for business in f:

            # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
            # First throwing away businesses that are not fit for our analysis
            business_dict = json.loads(business)
            # Check if business is restaurant
            categories = business_dict["categories"]
            if categories != None and any(x in categories for x in food_business_list):

                business_id = business_dict["business_id"]
                business_features = {}

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # If business is closed, only include those which closed during pandemic
                if business_dict["is_open"] != 1:
                    if business_id in checkin_dict.keys():
                        last_opened = checkin_dict[business_id]
                        if last_opened >= 2020:
                            business_features.update(
                                {"year_closed": last_opened})
                            closed_business_counter += 1
                        # Don't include business closed before pandemic
                    else:
                        continue
                else:
                    business_features.update({"year_closed": 0})

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # Check if business has attributes
                if business_dict["attributes"] == None:
                    continue

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # Print out progress
                if business_counter % 10000 == 1:
                    print("Loaded " + str(business_counter) + " businesses data")

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # Clean up features of the business, making indicators
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

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # Using Attributes to generate features
                business_attribute_list = []
                for attribute in business_dict["attributes"].keys():
                    business_attribute_list.append(attribute)

                if "RestaurantsPriceRange2" in business_attribute_list and business_dict["attributes"]["RestaurantsPriceRange2"] != "None":
                    business_features.update(
                        {"price_range": int(business_dict["attributes"]["RestaurantsPriceRange2"])})
                else:
                    business_features.update({"price_range": None})

                if "DietaryRestrictions" in business_attribute_list and business_dict["attributes"]["DietaryRestrictions"] != "None":
                    restriction = ast.literal_eval(
                        business_dict["attributes"]["DietaryRestrictions"])
                    restriction_ind = 0
                    for restriction_type in restriction.keys():
                        if restriction[restriction_type] == True:
                            restriction_ind = 1
                    business_features.update(
                        {"dietary_restriction": restriction_ind})
                else:
                    business_features.update({"dietary_restriction": None})

                if "BusinessParking" in business_attribute_list and business_dict["attributes"]["BusinessParking"] != "None":
                    parking = ast.literal_eval(
                        business_dict["attributes"]["BusinessParking"])
                    parking_ind = 0
                    for parking_type in parking.keys():
                        if parking[parking_type] == True:
                            parking_ind = 1
                    business_features.update({"parking": parking_ind})
                else:
                    business_features.update({"parking": None})

                # Features which are booleans which we take directly from the data
                data_features = ["RestaurantsTakeout", "DriveThru", "OutdoorSeating", "RestaurantsDelivery", "Caters", "RestaurantsReservations", "WheelChairAccessible", "HasTv", "DogsAllowed",
                                 "GoodForKids", "RestaurantsGoodForGroups", "RestaurantsTableService", "RestaurantsCounterService", "HappyHour", "BusinessAcceptsCreditCards", "BusinessAcceptsBitcoin", "AcceptsInsurance"]
                our_features = ["takeout", "drive_through", "outdoor_seating", "delivery", "caters", "reservations", "accessible", "has_tv", "dogs_allowed",
                                "good_for_kids", "good_for_groups", "table_service", "counter_service", "happy_hour", "accepts_credit_card", "accepts_bitcoin", "accepts_insurance"]
                counter = 0
                for feature in data_features:
                    if feature in business_attribute_list and business_dict["attributes"][feature] != "None":
                        if business_dict["attributes"][feature] == "True":
                            business_features.update(
                                {our_features[counter]: 1})
                        else:
                            business_features.update(
                                {our_features[counter]: 0})
                    else:
                        business_features.update({feature: None})
                    counter += 1

                no_alc = ["u'none'", "'none'"]
                yes_alc = ["'full_bar'", "u'beer_and_wine'",
                           "u'full_bar'", "'beer_and_wine'"]
                if "Alcohol" in business_attribute_list:
                    if business_dict["attributes"]["Alcohol"] in yes_alc:
                        business_features.update({"alcohol": 1})
                    elif business_dict["attributes"]["Alcohol"] in no_alc:
                        business_features.update({"alcohol": 0})
                    else:
                        business_features.update({"alcohol": None})

                if "AgesAllowed" in business_attribute_list:
                    if business_dict["attributes"]["AgesAllowed"] == "u'allages'":
                        business_features.update({"all_ages": 1})
                    elif business_dict["attributes"]["AgesAllowed"] == "u'21plus'":
                        business_features.update({"all_ages": 0})
                    else:
                        business_features.update({"all_ages": None})

                if "Wifi" in business_attribute_list:
                    if business_dict["attributes"]["Wifi"] == "u'no'" or business_dict["attributes"]["Wifi"] == "'no'":
                        business_features.update({"wifi_available": 0})
                        business_features.update({"wifi_paid": 0})
                    elif business_dict["attributes"]["Wifi"] == "u'free'" or business_dict["attributes"]["Wifi"] == "'free'":
                        business_features.update({"wifi_available": 1})
                        business_features.update({"wifi_paid": 0})
                    elif business_dict["attributes"]["Wifi"] == "u'paid'" or business_dict["attributes"]["Wifi"] == "'paid'":
                        business_features.update({"wifi_available": 1})
                        business_features.update({"wifi_paid": 1})
                    else:
                        business_features.update({"wifi_available": None})
                        business_features.update({"wifi_paid": None})

                if "NoiseLevel" in business_attribute_list:
                    if business_dict["attributes"]["NoiseLevel"] == "u'quiet'" or business_dict["attributes"]["NoiseLevel"] == "'quiet'":
                        business_features.update({"noise_level": 1})
                    elif business_dict["attributes"]["NoiseLevel"] == "u'average'" or business_dict["attributes"]["NoiseLevel"] == "'average'":
                        business_features.update({"noise_level": 2})
                    elif business_dict["attributes"]["NoiseLevel"] == "u'loud'" or business_dict["attributes"]["NoiseLevel"] == "'loud'":
                        business_features.update({"noise_level": 3})
                    elif business_dict["attributes"]["NoiseLevel"] == "u'very_loud'" or business_dict["attributes"]["NoiseLevel"] == "'very_loud'":
                        business_features.update({"noise_level": 4})
                    else:
                        business_features.update({"noise_level": None})
                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                # Using hours to create features
                open_days = 0
                for day in days_of_week:
                    if (business_dict["hours"] != None) and (day in business_dict["hours"].keys()):
                        d1, d2 = business_dict["hours"][day].split('-')
                        d1 = datetime.strptime(d1, '%H:%M')
                        d2 = datetime.strptime(d2, '%H:%M')
                        business_features.update(
                            {day: int((d2-d1).seconds / 3600)})
                        open_days += 1
                    else:
                        business_features.update({day: 0})
                # Summing up hours
                total = 0
                for day in days_of_week:
                    total += business_features[day]
                business_features.update({"total_hours": total})
                # check weekday/weekend
                wd = 0
                we = 0
                for weekday in weekdays:
                    if business_features[weekday] > 0:
                        wd = 1
                for weekend in weekends:
                    if business_features[weekend] > 0:
                        we = 1
                business_features.update({"weekday": wd})
                business_features.update({"weekend": we})
                business_features.update({"num_open_days": open_days})

                # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
                business_counter += 1
                business_list_lasso.append((business_id, business_features))
                business_list_data.append((business_id, business_dict))

    # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
    # for i in range(10):
    #     print(business_list[i]["name"])
        # print(business_list[i]["categories"])
    print("Businesses in filtered dataset: " + str(business_counter))
    print("Closed businesses in filtered dataset: " +
          str(closed_business_counter))
    print(business_list_lasso[100:150])
    # print(business_list_data[500][1]["attributes"]["BusinessParking"]["garage"])
    # print(len(business_list))
    f.close()
    return business_list_lasso, business_list_data

def load_review_data(path, business_list_lasso, business_list_data):
    print("Reading in Review JSON File")
    with open(path) as f:
        for review in f:
            print(review)
        # # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
        #     # First throwing away businesses that are not fit for our analysis
        #     review_dict = json.loads(review)

    return business_list_lasso, business_list_data

def main(args):
    if args.readbusinessdata:
        checkin_dict = load_checkin_data(
            "yelp_dataset/yelp_academic_dataset_checkin.json")
        business_list_lasso, business_list_data = load_business_data(
            "yelp_dataset/yelp_academic_dataset_business.json", checkin_dict)
        with open('business_list_lasso', 'wb') as file:
            pickle.dump(business_list_lasso, file)
        with open('business_list_data', 'wb') as file:
            pickle.dump(business_list_data, file)
    else:
        # If not reading in / processing data again, load the data
        with open('business_list_lasso', 'rb') as file:
            business_list_lasso = pickle.load(file)
        with open('business_list_data', 'rb') as file:   
            business_list_data = pickle.load(file)
    
    if args.readreviewdata:
        business_list_lasso_reviewed, business_list_data_reviewed = load_review_data("yelp_dataset/yelp_academic_dataset_review.json", business_list_lasso, business_list_data)
        with open('business_list_lasso_reviewed', 'wb') as file:
            pickle.dump(business_list_lasso_reviewed, file)
        with open('business_list_data_reviewed', 'wb') as file:
            pickle.dump(business_list_data_reviewed, file)
    else:
        with open('business_list_lasso_reviewed', 'rb') as file:
            business_list_lasso_reviewed = pickle.load(file)
        with open('business_list_data_reviewed', 'rb') as file:
            business_list_data_reviewed = pickle.load(file)



if __name__ == '__main__':
    args = parseArguments()
    main(args)
