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
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import Lasso
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


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
                'Washington', 'Dallas', 'Houston', 'Boston', 'Philadelphia', 'Atlanta', 'Seattle', 'San Jose']
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
    parser.add_argument("-fd", "--formatdata",
                        help="Format Data for Lasso", action="store_true")
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
    business_list_lasso = {}
    business_list_data = {}
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
                            business_dict.update(
                                {"year_closed": last_opened})
                            closed_business_counter += 1
                        else:
                            continue
                        # Don't include business closed before pandemic
                    else:
                        continue
                else:
                    business_dict.update({"year_closed": 0})

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
                business_attribute_list = list(business_dict["attributes"].keys())

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
                data_features = ["RestaurantsTakeOut", "DriveThru", "OutdoorSeating", "RestaurantsDelivery", "Caters", "RestaurantsReservations", "WheelchairAccessible", "HasTV", "DogsAllowed",
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
                        business_features.update({our_features[counter]: None})
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
                else:
                    business_features.update({"alcohol": None})

                if "AgesAllowed" in business_attribute_list:
                    if business_dict["attributes"]["AgesAllowed"] == "u'allages'":
                        business_features.update({"all_ages": 1})
                    elif business_dict["attributes"]["AgesAllowed"] == "u'21plus'":
                        business_features.update({"all_ages": 0})
                    else:
                        business_features.update({"all_ages": None})
                else:
                    business_features.update({"all_ages": None})

                if "WiFi" in business_attribute_list:
                    if business_dict["attributes"]["WiFi"] == "u'no'" or business_dict["attributes"]["WiFi"] == "'no'":
                        business_features.update({"wifi_available": 0})
                        business_features.update({"wifi_paid": 0})
                    elif business_dict["attributes"]["WiFi"] == "u'free'" or business_dict["attributes"]["WiFi"] == "'free'":
                        business_features.update({"wifi_available": 1})
                        business_features.update({"wifi_paid": 0})
                    elif business_dict["attributes"]["WiFi"] == "u'paid'" or business_dict["attributes"]["WiFi"] == "'paid'":
                        business_features.update({"wifi_available": 1})
                        business_features.update({"wifi_paid": 1})
                    else:
                        business_features.update({"wifi_available": None})
                        business_features.update({"wifi_paid": None})
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
                business_list_lasso.update({business_id:business_features})
                business_list_data.update({business_id: business_dict})

    # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
    # for i in range(10):
    #     print(business_list[i]["name"])
        # print(business_list[i]["categories"])
    print("Businesses in filtered dataset: " + str(business_counter))
    print("Closed businesses in filtered dataset: " +
          str(closed_business_counter))
    print(business_list_lasso[list(business_list_lasso.keys())[4]])
    # print(business_list_data[500][1]["attributes"]["BusinessParking"]["garage"])
    # print(len(business_list))
    f.close()
    return business_list_lasso, business_list_data

def load_review_data(path, business_list_lasso, business_list_data):
    print("Reading in Review JSON File")

    # Instantiate new SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    temp_business_stats = {}
    for business in business_list_lasso.keys():
        temp_business_stats.update({business : {}})
        temp_business_stats[business].update({"total_reviews":0})
        temp_business_stats[business].update({"total_stars":0})
        temp_business_stats[business].update({"total_words":0})
        temp_business_stats[business].update({"neg":0})
        temp_business_stats[business].update({"neu":0})
        temp_business_stats[business].update({"pos":0})
        temp_business_stats[business].update({"compound":0})

    review_stats = {}
    years = [2018, 2019, 2020, 2021, 2022]
    for year in years:
        review_stats.update({year:0})

    with open(path) as f:
        counter = 0
        counted_reviews = 0
        useful_counter = 0
        funny_counter = 0
        cool_counter = 0
        for review in f:
        # ♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡･･*･･♡ #
            review_dict = json.loads(review)
            counter += 1
            if counter > 10000:
                break
            business_id = review_dict["business_id"]
            # Check if review is for a business in our filtered dataset
            if business_id in business_list_lasso.keys():
                counted_reviews += 1
                 # Print out progress
                if counted_reviews % 1000 == 1:
                    print("Loaded " + str(counted_reviews) + " businesses data")
                temp_dict = temp_business_stats[business_id]
                temp_dict.update({"total_reviews":temp_dict["total_reviews"] + 1})

                stars = review_dict["stars"]
                temp_dict.update({"total_stars":temp_dict["total_stars"] + stars})

                useful_counter += review_dict["useful"]
                funny_counter += review_dict["funny"]
                cool_counter += review_dict["cool"]

                date = datetime.strptime(review_dict["date"], '%Y-%m-%d %H:%M:%S')
                if date.year in years:
                    review_stats.update({date.year: review_stats[date.year] + 1})

                review_text = review_dict["text"]
                tokenizer = RegexpTokenizer(r'\w+')
                count = len(tokenizer.tokenize(review_text))
                temp_dict.update({"total_words":temp_dict["total_words"] + count})
                sentiment_dict = sid.polarity_scores(review_text)
                temp_dict.update({"neg":temp_dict["neg"] + (sentiment_dict["neg"]*100)})
                temp_dict.update({"neu":temp_dict["neu"] + (sentiment_dict["neu"]*100)})
                temp_dict.update({"pos":temp_dict["pos"] + (sentiment_dict["pos"]*100)})
                temp_dict.update({"compound":temp_dict["compound"] + (sentiment_dict["compound"]*100)})

    for business in business_list_lasso.keys():
        temp_dict = temp_business_stats[business]
        if temp_dict["total_reviews"] != 0:
            avg_rev_len = temp_dict["total_words"] / temp_dict["total_reviews"]
            avg_neg = temp_dict["neg"] / temp_dict["total_reviews"]
            avg_neu = temp_dict["neu"] / temp_dict["total_reviews"]
            avg_pos = temp_dict["pos"] / temp_dict["total_reviews"]
            avg_compound = temp_dict["compound"] / temp_dict["total_reviews"]
            avg_stars = temp_dict["total_stars"] / temp_dict["total_reviews"]
        else:
            avg_rev_len = 0
            avg_neg = 0
            avg_neu = 0
            avg_pos = 0
            avg_compound = 0
            avg_stars = 0

        business_list_lasso[business].update({"avg_review_length":avg_rev_len})
        business_list_lasso[business].update({"avg_negative_score":avg_neg})
        business_list_lasso[business].update({"avg_neutral_score":avg_neu})
        business_list_lasso[business].update({"avg_positive_score":avg_pos})
        business_list_lasso[business].update({"avg_compound_score":avg_compound})
        business_list_data[business].update({"counted_reviews":temp_dict["total_reviews"]})
        business_list_data[business].update({"counted_avg_stars":avg_stars})


    avg_useful = useful_counter / counted_reviews
    avg_cool = cool_counter / counted_reviews
    avg_funny = funny_counter / counted_reviews
    review_stats.update({"counted_reviews" :counted_reviews})
    review_stats.update({"avg_useful":avg_useful})
    review_stats.update({"avg_cool":avg_cool})
    review_stats.update({"avg_funny":avg_funny})

    for i in range(1):
        rand = int(random.randint(0,1000))
        print(rand)
        print(business_list_lasso[list(business_list_lasso.keys())[rand]])
        # print(business_list_data[list(business_list_lasso.keys())[rand]])
    print(review_stats)

    return business_list_lasso, business_list_data, review_stats

def read_city_data(path, business_list_lasso, cities_list):
    filtered_business= {}
    # Reading in CSV file
    with open(path) as cities_file:
        csvreader = csv.DictReader(cities_file)
        headers = csvreader.fieldnames
        city_dict = {} # {City: {Dictionary of features of city}}
 
        # for city in cities_list:
        for row in csvreader: # loops through a city
            features_dict = row
            city_dict.update({features_dict["city"]: features_dict}) # saves dictionary to the cities dictionary: { LA: {GDP: 10000, Precipitation: 3, etc.}}
    
    # Make features for each restaurant based on city they are in
    restaurants = list(business_list_lasso.keys())
    for restaurant in restaurants:
        for city in cities_list: # loops through possible cities
            restaurant_features = business_list_lasso[restaurant]
            if restaurant_features[city] == 1: # checks relevant city
                restaurant_features.update(city_dict[city])
                restaurant_features.pop("city")
                # print(restaurant_features)
                #print("RESTAURANT FEATURES")
               # print(restaurant_features)
                #filtered_business.update({restaurant:business_list_lasso[restaurant]}) # merges dictionary
                filtered_business.update({restaurant:restaurant_features}) # merges dictionary

    return filtered_business

def reformat_data(business_list_lasso_reviewed):
# Redundant code to reformat data ... sad....
    print("Reformatting Data")
    X = []
    y = []
    counter = 1
    businesses = list(business_list_lasso_reviewed.keys())
    features = business_list_lasso_reviewed[businesses[0]].keys()
    for business in businesses:
        data = []
        for feature in features:
            if feature != 'is_open':
                # if feature == "year_closed":
                    # print(business_list_lasso_reviewed[business])
                data.append(business_list_lasso_reviewed[business][feature])
            else:
                y.append(business_list_lasso_reviewed[business][feature])
        X.append(data)
    features = [x for x in features if x != "is_open"]
    # Make Imputations to data before lasso-ing
    print("Performing Imputations")
    # imputer = KNNImputer(n_neighbors=5, weights="uniform")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value = 0)
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    output_features = imputer.get_feature_names_out(input_features=features)
    for feat in features:
        if feat not in output_features:
            print(feat)
    
    return X, y, features

def lasso_reg(X, y, alpha, features):
    print("Performing Lasso")
    model = Lasso(alpha)
    model.fit(X,y)
    betas = model.coef_
    constant = model.intercept_
    result = []
    result.append(["constant", constant])
    # print(len(features))
    print(len(betas))
    for i in range(len(features)-1):
        result.append([features[i], betas[i]])
    return result

def ridge_reg(X, y, alpha, features):
    print("Performing Lasso")
    model = Ridge(alpha)
    model.fit(X,y)
    betas = model.coef_
    constant = model.intercept_
    result = []
    result.append(["constant", constant])
    # print(len(features))
    print(len(betas))
    for i in range(len(features)-1):
        result.append([features[i], betas[i]])
    return result

def check_business_data(attribute, data):
    yes_count = 0
    no_count = 0
    for business in list(data.keys()):
        if data[business][attribute] == 1:
            yes_count += 1
        else:
            no_count += 1
    print("yes" + str(yes_count))
    print("no" + str(no_count))

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
    
    check_business_data("wifi_available", business_list_lasso)
    
    if args.readreviewdata:
        business_list_lasso_reviewed, business_list_data_reviewed, review_stats = load_review_data("yelp_dataset/yelp_academic_dataset_review.json", business_list_lasso, business_list_data)
        with open('business_list_lasso_reviewed', 'wb') as file:
            pickle.dump(business_list_lasso_reviewed, file)
        with open('business_list_data_reviewed', 'wb') as file:
            pickle.dump(business_list_data_reviewed, file)
        with open('review_stats', 'wb') as file:
            pickle.dump(review_stats, file)
    else:
        with open('business_list_lasso_reviewed', 'rb') as file:
            business_list_lasso_reviewed = pickle.load(file)
        with open('business_list_data_reviewed', 'rb') as file:
            business_list_data_reviewed = pickle.load(file)

    if args.formatdata:
        X, y, features = reformat_data(business_list_lasso_reviewed)
        X_c, y_c, features_c = reformat_data(read_city_data("1660 final project extra features - the whole sheet_.csv", business_list_lasso_reviewed, major_cities))
        with open('X_formatted', 'wb') as file:
            pickle.dump(X, file)
        with open('y_formatted', 'wb') as file:
            pickle.dump(y, file)
        with open('features', 'wb') as file:
            pickle.dump(features, file)
        with open('X_formatted_c', 'wb') as file:
            pickle.dump(X_c, file)
        with open('y_formatted_c', 'wb') as file:
            pickle.dump(y_c, file)
        with open('features_c', 'wb') as file:
            pickle.dump(features_c, file)
    else:
        with open('X_formatted', 'rb') as file:
            X = pickle.load(file)
        with open('y_formatted', 'rb') as file:
            y = pickle.load(file)
        with open('features', 'rb') as file:
            features = pickle.load(file)
        with open('X_formatted_c', 'rb') as file:
            X_c = pickle.load(file)
        with open('y_formatted_c', 'rb') as file:
            y_c = pickle.load(file)
        with open('features_c', 'rb') as file:
            features_c = pickle.load(file)
    
    print("Lasso on entire filtered dataset")
    result = lasso_reg(X, y, 0.01, features)
    print(result)
    print("Lasso on filtered city dataset")
    result_c = lasso_reg(X_c, y_c, 0.01, features_c)
    print(result_c)

    print("Ridge on entire filtered dataset")
    result = ridge_reg(X, y, 0.01, features)
    print(result)
    print("Ridge on filtered city dataset")
    result_c = ridge_reg(X_c, y_c, 0.01, features_c)
    print(result_c)



if __name__ == '__main__':
    args = parseArguments()
    main(args)
