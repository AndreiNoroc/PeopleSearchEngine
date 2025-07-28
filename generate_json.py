import json
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
import nltk
import os
from langdetect import detect
# import requests
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from gensim.models import KeyedVectors
import copy
import numpy as np

model_path = "path\\GoogleNews-vectors-negative300.bin"
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def get_ngrams(raw_text):
    def generate_ngrams(text, n):
        tokens = word_tokenize(text.lower())
        return list(ngrams(tokens, n))

    bigrams = generate_ngrams(raw_text, 2)
    trigrams = generate_ngrams(raw_text, 3)

    bigrams = [" ".join(gram) for gram in bigrams]
    trigrams = [" ".join(gram) for gram in trigrams]

    return bigrams + trigrams

def get_keywords(raw_text):
    # url = "https://api.cortical.io/nlp/keywords"
    # headers = {
    #     "Content-Type": "application/json"
    # }
    # data = {
    #     "text": raw_text,
    #     "language": "en"
    # }
    # get_keywords = requests.post(url, json=data, headers=headers)

    # if get_keywords.status_code == 200:
    #     print("Request successful")
    #     get_keywords = get_keywords.json()
    #     return [entry['word'] for entry in get_keywords['keywords']]
    # else:
    #     print("Request failed with status code:", get_keywords)
    #     print("Response body:", get_keywords)

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(raw_text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    return filtered_words[:10]

def send_json_to_opensearch(json_person):
    host = [{'host': 'search-people-search-sri-u5d3tvbw5zb3ut5xiv47b4jcqe.aos.eu-north-1.on.aws', 'port': 443}]
    auth = ('user', 'pass')

    client = OpenSearch(
        hosts=host,
        http_compress=True,
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    response = client.index(index='people_search_v2', body=json_person)
    print("DONE POST")

    response = response.json()

    if response['result'] == 'created':
        print("Document indexed successfully:", response.json())
        return response
    else:
        print("Failed to index document:", response)

    return None

def get_json_profile_from_opensearch(profile_name):
    host = [{'host': 'search-people-search-sri-u5d3tvbw5zb3ut5xiv47b4jcqe.aos.eu-north-1.on.aws', 'port': 443}]
    auth = ('user', 'pass')

    client = OpenSearch(
        hosts=host,
        http_compress=True,
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    search_query = {
        "size": 1,
        "query": {
            "match": {
                "name": profile_name
            }
        },
        "sort": [
            {
                "version": {
                    "order": "desc"
                }
            }
        ]
    }

    response = client.search(index='people_search_v2', body=search_query)

    return response["hits"]["hits"][0]["_source"]
    
# sentence transformers improvments
def transform_text_to_vec_embeddings(raw_text):
    tokens = raw_text.lower().split()
    selected_embeddings = [word2vec[word] for word in tokens if word in word2vec]
    mean_embedding = np.mean(selected_embeddings, axis=0)

    return mean_embedding.tolist()

def read_data_person_and_normalize_it(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                #old_profile = copy.deepcopy(get_json_profile_from_opensearch(data['name']))

                social_media_profiles = data['social_media_profiles']

                # if old_profile != None:
                #     social_media_profiles = old_profile['social_media_profiles'] + data['social_media_profiles']

                social_media_profiles = data['social_media_profiles']    

                social_media_profiles = list(set(social_media_profiles))

                name = data['name']

                version = 1
                # if old_profile != None:
                #     version = old_profile['version'] + 1

                keywords = []
                # if old_profile != None:
                #     keywords = old_profile['keywords']

                ngrams = []
                # if old_profile != None:
                #     ngrams = old_profile['n_grams']

                emb_vector = []
                # if old_profile != None:
                #     emb_vector = old_profile['vector_embedding']

                entire_text_posts = ""
                for obj in data['content']:
                    get_text = obj['text']
                    entire_text_posts = entire_text_posts + get_text
                    # extract language and verify it
                    language = detect(get_text)
                    if  language != "en":
                        continue

                    # extract keywords
                    keywords.extend(get_keywords(get_text))
                    keywords = list(set(keywords))

                    # extract n-grams
                    ngrams.extend(get_ngrams(get_text))
                    ngrams = list(set(ngrams))

                    # preprocess words to embedding vectors
                    emb_vector.extend(transform_text_to_vec_embeddings(get_text))
                    emb_vector = list(set(emb_vector))

                timestamp = int(datetime.now().timestamp()) * 1000

                json_normalized_data = {
                    "name": name,
                    "social_media_profiles": social_media_profiles,
                    "keywords": keywords,
                    "n_grams": ngrams,
                    "vector_embedding": emb_vector,
                    "timestamp": timestamp,
                    "version": version
                }

                # file_path = "data.json"

                # with open(file_path, "w") as json_file:
                #     json.dump(json_normalized_data, json_file, indent=4)

                # resp = send_json_to_opensearch(json_normalized_data)
                print("DONE\n" + entire_text_posts)
                # print(resp)

if __name__ == "__main__":
    print("<--------------------Welcome to the party-------------------->")
    read_data_person_and_normalize_it("path\\people_search\\people_profiles\\")