#!/usr/bin/env python
# coding: utf-8

# In[176]:


import json
from collections import defaultdict
# List to store data from all files
combined_data = []

# Loop through file indices
lastN=50
filename = "New_York_restaurant_aspects_compress_range_"
filename = "New_York_restaurant_aspects_range_"
   
#print(len(data))
#print(len(combined_dict))
with open(filename+f"0to{50*lastN}.json", 'r') as file:
    total_data = json.load(file)
#print(len(total_data))


# In[177]:


#%pip install inflect

#"""
#combined_dict = defaultdict()
#for i in range(lastN):  # 11 because the range goes from 0 to 10 inclusive
#    temp_file_name = filename+f"{50*i}to{50*(i+1)}.json"
#    with open(temp_file_name, 'r') as file:
#        # Load data from current file and add it to the combined list
#        data = json.load(file)
#       #combined_data.append(data)
#        for k, v in data.items():
#            combined_dict[k] = v
            
            

# Write the combined data to a new file
#with open(filename+f"0to{50*lastN}.json", 'w') as file:
#    json.dump(combined_dict, file, indent=4)
#"""; 
import inflect

def to_singular(word):
    p = inflect.engine()
    
    # Check if the word is plural and return the singular form if so
    singular = p.singular_noun(word)
    if singular:
        return singular
    else:
        return word  # Return the original word if it is not plural

# Test the function
#print(to_singular("apples"))  # Should return "apple"
#len(combined_dict)


# In[178]:


#k = list(total_data)
#data0 = {k[0]:total_data[k[0]]}
#print(data0)
from collections import defaultdict
import contextlib
import sys
import io
import json
#from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints
#checkpoint_map = available_checkpoints()

#from pyabsa import ABSAInstruction
#type(businessrevdata)
#reviews = list(businessrevdata.values())
#len(reviews)
#len(reviews[1])
#generator = ABSAInstruction.ABSAGenerator("multilingual")
#"""
#aspect_extractor = ATEPC.AspectExtractor('multilingual',
#                                         auto_device=True,  # False means load model on CPU
#                                         cal_perplexity=True,
#                                         )
#"""
def flatten_list(list):
    flat_list = []
    [flat_list.append(nest_l) for l in list for nest_l in l]
    return flat_list

def convert_sentiment(list):
    lnum = []
    for index, l in enumerate(list):
        if l=='Positive':
            lnum.append(1)
        elif l=='Negative':
            lnum.append(-1)        
        else:
            lnum.append(0)
    return lnum
def aspectExtractor(businessrevdata, keycnt1=0, keycnt2=0, generator=None, filename=''):
    pyabsa_out = defaultdict()
    Aspects = defaultdict()    
    Aspects_compress = defaultdict()    
    keycnt = keycnt1;
    if not generator:
        generator = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         print_result=False,                                         
                                         )          
    if keycnt2==0:
        keycnt2 = len(businessrevdata)
    klist = list(businessrevdata)
    #for k,v in businessrevdata.items():
    for i in range(keycnt1, keycnt2):
        k = klist[i]
        v = businessrevdata[k]
        if keycnt>keycnt2:
            break
        
        print(f"keycnt = {keycnt}\n")
        
        #print(f"v={v}")
        #print(reviews[20][i], "\n")        
        revcnt = 0;
        #print(f"len = {len(v['reviews'])}")
        for rev in v['reviews'][0]:
            #result =
            if revcnt%10==0:
                print(f"revcnt={(revcnt)}")
            if revcnt>30:
                break
            #    break            
            revcnt += 1
            #q=generator.predict(rev)['Quadruples']


            original_stdout = sys.stdout

            # Block of code where you want to suppress the output
            #sys.stdout = io.StringIO()
            # End of block            
            with contextlib.redirect_stdout(io.StringIO()):
                q=generator.predict(rev, save_result=False, print_result=False)
            # Restore the original stdout
            #sys.stdout = original_stdout
            print(f"len(q) = {len(q)}")
            #for nq in range(len(q)):
            print(f"q={q}")
            aspect = q['aspect']
            sentiment = q['sentiment']
            probs = q['probs']
            confidence = q['confidence']
            if len(aspect)>0:
                if k not in Aspects.keys():
                    
                    Aspects[k] = defaultdict(list)
                    Aspects_compress[k] = defaultdict(list)
                    
                    Aspects[k]['name'] = businessrevdata[k]['name']                    
                    Aspects[k]['address'] = businessrevdata[k]['address']                    
                    Aspects[k]['latitude'] = businessrevdata[k]['latitude']
                    Aspects[k]['longitude'] = businessrevdata[k]['longitude']
                    Aspects[k]['avg_rating'] = businessrevdata[k]['avg_rating']
                    Aspects[k]['num_of_reviews'] = businessrevdata[k]['num_of_reviews']
                    Aspects[k]['url'] = businessrevdata[k]['url']
                    Aspects[k]['hours'] = businessrevdata[k]['hours']
                    Aspects[k]['category'] = businessrevdata[k]['category']                    

                    Aspects_compress[k]['name'] = businessrevdata[k]['name']                    
                    Aspects_compress[k]['address'] = businessrevdata[k]['address']                    
                    Aspects_compress[k]['latitude'] = businessrevdata[k]['latitude']
                    Aspects_compress[k]['longitude'] = businessrevdata[k]['longitude']
                    Aspects_compress[k]['avg_rating'] = businessrevdata[k]['avg_rating']
                    Aspects_compress[k]['num_of_reviews'] = businessrevdata[k]['num_of_reviews']
                    #Aspects_compress[k]['hours'] = businessrevdata[k]['hours']
                    Aspects_compress[k]['category'] = businessrevdata[k]['category']                    

                    
                    Aspects[k]['aspects'].append(aspect)
                    Aspects[k]['sentiment'].append(sentiment)
                    Aspects[k]['probs'].append(probs)
                    Aspects[k]['confidence'].append(confidence)
                else:
                    Aspects[k]['aspects'].append(aspect)
                    Aspects[k]['sentiment'].append(sentiment)
                    Aspects[k]['probs'].append(probs)
                    Aspects[k]['confidence'].append(confidence)
        if k in Aspects.keys():           
            Aspects[k]['aspects'] = flatten_list(Aspects[k]['aspects'])
            Aspects[k]['sentiment'] = flatten_list(Aspects[k]['sentiment'])
            Aspects[k]['sentiment'] = convert_sentiment(Aspects[k]['sentiment'])
            Aspects[k]['probs'] = flatten_list(Aspects[k]['probs'])
            Aspects[k]['confidence'] = flatten_list(Aspects[k]['confidence'])
            Aspects_compress[k]['aspects'] = Aspects[k]['aspects']
            Aspects_compress[k]['sentiment'] = Aspects[k]['sentiment']
        keycnt += 1        
        if keycnt%1000==0 or keycnt==keycnt2:
            with open(f"{filename}_aspects_range_{keycnt1}to{keycnt}.json", "w") as fp:
                json.dump(Aspects, fp)
            with open(f"{filename}_aspects_compress_range_{keycnt1}to{keycnt}.json", "w") as fp:
                json.dump(Aspects_compress, fp)        
                #aspect = q[nq]['aspect']
                #polarity = q[nq]['polarity']
                #aspect = q[nq]['aspect']
                #print(f"aspect = {aspect}")
                #print(f"polarity = {polarity}, \n")
                #Aspects[k].append(tuple((aspect, polarity)))
                #print(result, "\n")
                #print(f"type = {type(result)}")
                #print(f"rev={rev}, \n")
        
        print(f"\n\n\n\n\n")
    return Aspects, Aspects_compress
        
#extracted_aspects = aspectExtractor(businessrevdata, generator)
#with open("total_restaurant_businessdata.json", "r") as fp:
#    total_reviews = json.load(fp)

#for i in range(0, 50):
#    extracted_aspects, comp_extracted_aspects = aspectExtractor(total_reviews, 50*i, 50*(i+1), None, 'New_York_restaurant')


# In[196]:


import copy
from collections import defaultdict
def combine_aspects(data):
    new_data = copy.deepcopy(data)
    for k,v in new_data.items():
        new_v = v.copy()
        aspects = v['aspects']
        sentiments = v['sentiment']        
        confidences = v['confidence']
        combined_aspects = []
        combined_sentiment = []
        combined_confidences = []
        sentiments_counter = []
        for cnt, asp in enumerate(aspects):            
            lower_asp = asp.lower()
            if '\n' in asp:
                #print('break')
                continue
                #break
            #print(f"asp={asp}, lenasp={len(lower_asp)}, lower_asp = {lower_asp}")
            #if asp=='\n\n\n':
            #    print(f"asp \n\n\n={asp}")
            #
            singular_asp = to_singular(lower_asp)
            if singular_asp not in combined_aspects:
                combined_aspects.append(singular_asp)
                combined_sentiment.append(sentiments[cnt])
                combined_confidences.append(confidences[cnt])
                sentiments_counter.append(1)
            else:
                index_asp = combined_aspects.index(singular_asp)
                combined_sentiment[index_asp] += sentiments[cnt]
                combined_confidences[index_asp] += confidences[cnt]
                sentiments_counter[index_asp] += 1
                #combined_confidences[index_asp] = combined_confidences[index_asp]/len()        
        new_data[k]['aspects'] = combined_aspects
        new_data[k]['sentiment'] = combined_sentiment
        new_data[k]['counter'] = sentiments_counter
        new_data[k]['confidence'] = combined_confidences        
        del new_data[k]['probs']
    return new_data               

#combine_data0 = combine_aspects(data0)
#combine_data = combine_aspects(total_data)


# In[171]:


#type(combine_data)
#combine_data[list(combine_data)[0]]


# In[197]:


#with open(f"total_combined_restaurant_businessdata_0to{50*lastN}.json", "w") as fp:
#    json.dump(combine_data, fp)
with open(f"total_combined_restaurant_businessdata_0to{50*lastN}.json", "r") as fp:
    combine_data = json.load(fp)    
#len(combine_data)


# In[181]:


def flatten(list):
    return [i for l in list for i in list]    


# In[182]:


data = {
    'gmap_id1': {'aspects': ['cleanliness', 'service'], 'ratings': [5, 4]},
    'gmap_id2': {'aspects': ['cleanliness', 'location'], 'ratings': [3, 5]},
    'gmap_id3': {'aspects': ['service', 'location'], 'ratings': [4, 3]}
}
import heapq

def find_top_rataings_gmap_ids(data, aspect, top_n=5):
    ratings_heap = []  # Using a min heap to keep track of top ratings
    #print(f"aspect= {aspect}")
    aspect = aspect.lower()
    aspect = to_singular(aspect)
    #print(f"aspect2= {aspect}")
    for gmap_id, details in data.items():
        if aspect in details['aspects']:
            # Find the index of the aspect to get the corresponding rating
            index = details['aspects'].index(aspect)
            rating = details['sentiment'][index]
            #print(f"rating={rating}")
            # Use a heap of size top_n to store the top ratings
            if len(ratings_heap) < top_n:
                heapq.heappush(ratings_heap, (rating, gmap_id))
            else:
                # If the current rating is higher than the smallest in the heap, replace it
                heapq.heappushpop(ratings_heap, (rating, gmap_id))


    # Extract the gmap_ids from the heap, they will be in ascending order of ratings
    #top_ratings_gmap_ids = [gmap_id for _, gmap_id in sorted(ratings_heap, reverse=True)]
    print(sorted(ratings_heap, reverse=True))
    #gmapandrating = [(data[gmap_id], r) for r, gmap_id in sorted(ratings_heap, reverse=True)]
    ratings = [r for r, _ in sorted(ratings_heap, reverse=True)]
    #print(ratings)
    gmap = [data[gmap_id] for _, gmap_id in sorted(ratings_heap, reverse=True)]
    #ratings = [data[gmap_id]['sentiment'] for _, gmap_id in sorted(ratings_heap, reverse=True)]
    location = [(data[gmap_id]['latitude'],data[gmap_id]['longitude'])  for _, gmap_id in sorted(ratings_heap, reverse=True)]
    name = [data[gmap_id]['name']  for _, gmap_id in sorted(ratings_heap, reverse=True)]
    address = [data[gmap_id]['address']  for _, gmap_id in sorted(ratings_heap, reverse=True)]
    #print(f"top_ratings_location = {top_ratings_location}")
    result = []
    for cnt in range(len(gmap)):
        result.append((name[cnt], {'lat': location[cnt][0], 'lon': location[cnt][1], \
                                          'address':address[cnt], 'rating':ratings[cnt]})) #, 'rating':ratings[cnt]}))
    return result
# Example usage
#desired_aspect = 'cleanliness'
#top_gmap_ids = find_top_rataings_gmap_ids(data, desired_aspect, top_n=5)
#print(top_gmap_ids)


# In[183]:


#desired_aspect = 'breakfast'
#toprating = find_top_rataings_gmap_ids(combine_data, desired_aspect, top_n=5)
#for k in range(len(toprating)):
    #print(k)
#    desired_aspect = to_singular(desired_aspect)
#    index=toprating[k]['aspects'].index(desired_aspect)    
#    rating = toprating[k]['sentiment'][index]
#    print(f"rating={rating}")
#"""; 
#desired_aspect = 'Breakfast'
#results = find_top_rataings_gmap_ids(combine_data, desired_aspect, top_n=5);
#results


# In[200]:


import streamlit as st
import pandas as pd

# Sample data: Dictionary with keywords each associated with latitude, longitude, and text
locations = {
    "New York": {"lat": 40.5, "lon": -74.2, "text": "The Big Apple"},
    "New York": {"lat": 40.65, "lon": -74.15, "text": "The Big Pie"},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "text": "City of Angels"},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "text": "The Windy City"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "text": "Space City"},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740, "text": "Valley of the Sun"},
    "Philadelphia": {"lat": 39.9526, "lon": -75.1652, "text": "The City of Brotherly Love"},
    "San Antonio": {"lat": 29.4241, "lon": -98.4936, "text": "Alamo City"},
    # Add more locations as needed
}

def search_locations(keyword, top_n=5):
    # Find matching locations and limit to top N results
    matches = {k: v for k, v in locations.items() if keyword.lower() in k.lower()}
    #print(f"matches = {matches}")
    return sorted(matches.items(), key=lambda x: x[0])[:top_n]

# Streamlit app
intro = "    Welcome to our innovative platform, where we specialize in aspect-based sentiment analysis to transform the way you discover restaurants through Google reviews for New York. In the vast sea of dining options and customer opinions, finding the perfect restaurant that matches your specific preferences can be overwhelming. Our advanced technology delves into the nuances of customer feedback, dissecting reviews to evaluate sentiments related to distinct aspects of dining experiences, such as food quality, service, ambiance, and more. By focusing on these details, we provide personalized restaurant recommendations that align perfectly with your desired dining feature, be it gourmet cuisine, exceptional service, or a cozy atmosphere. \
\n\n\
    Utilizing cutting-edge natural language processing algorithms, our system meticulously analyzes and interprets the sentiments expressed in Google reviews, offering you a curated list of restaurants that excel in the aspect you value most. Whether you're craving the best sushi in town or seeking a place with an enchanting view, our platform simplifies your search and guides you to the ideal spot. Alongside each recommendation, you'll find essential details including the restaurant's address and its precise location on the map, making your dining adventure effortless and enjoyable. Embark on a gastronomic journey with us, where your preferences lead the way to exceptional culinary experiences."

def main():
    st.title("Aspect-Based Restaurant Recommender")    
    # User input
    st.write(intro)
    desired_aspect = st.text_input("Enter your desired aspect:")
    #top_n = st.text_input("Enter how many recommendations you need:")
    #top_n = int(float(top_n))
    #if top_n>10 or top_n<0:
    top_n = 5
        
    if desired_aspect:
        # Search for locations matching the keyword
        results = \
            find_top_rataings_gmap_ids(combine_data, desired_aspect, top_n)
        #results = search_locations(keyword)
        print(f"len results = {results}")   
        if results:
            st.header('Here are our recommendations:')
            cnt = 1
            for name, info in results:
                
                st.subheader(f"{(cnt)} {name}")
                st.write(f"Name = {name}, Address={info['address']}, \
                    Total aspect-based positive reviews:{info['rating']}")
                cnt += 1
                #st.write(info['address'])
                #st.write(info['rating'])
                # Show on map     
            #locations = [results[lat],results[] for _, info in results]
            #st.map(pd.DataFrame(results, columns=['lat', 'lon']))            
            locations = [(info["lat"],info['lon']) for _, info in results]
            st.map(pd.DataFrame(locations, columns=['lat', 'lon']))
        else:
            st.error("No matching locations found. Please try a different keyword.")

if __name__ == "__main__":
    main()
#!jupyter nbconvert --to script PyabsaApp.ipynb
#search_locations('New York', top_n=5)


# In[201]:


#print(intro)


# In[186]:


#results = \
#            find_top_rataings_gmap_ids(combine_data, 'breakfast', top_n=5)


# In[194]:


#locations = [(info["lat"],info['lon']) for _, info in results]


# In[192]:


#locations


# In[193]:


#st.map(pd.DataFrame(locations, columns=['lat', 'lon']))


# In[ ]:




