# PeopleSearchEngine

## Overview
The People Search Engine is a tool designed to process and normalize data about individuals, generating structured JSON profiles that can be indexed and searched efficiently. It leverages natural language processing (NLP) techniques, vector embeddings, and OpenSearch for storing and retrieving profiles.

## JSON Generation Process
The `generate_json.py` script is responsible for generating JSON profiles from raw data. Below is an overview of its functionality:

1. **Data Reading and Normalization**:
   - The script reads JSON files from a specified folder containing raw data about individuals.
   - It processes the data to extract key information such as social media profiles, keywords, n-grams, and vector embeddings.

2. **Natural Language Processing**:
   - Extracts keywords using tokenization and stopword removal.
   - Generates bigrams and trigrams from the text content.
   - Converts text into vector embeddings using a pre-trained Word2Vec model.

3. **Language Detection**:
   - Ensures that only English content is processed by detecting the language of the text.

4. **JSON Structure**:
   - The normalized data is structured into a JSON format with the following fields:
     - `name`: The individual's name.
     - `social_media_profiles`: A list of unique social media profiles.
     - `keywords`: Extracted keywords from the content.
     - `n_grams`: Generated bigrams and trigrams.
     - `vector_embedding`: Text embeddings for semantic search.
     - `timestamp`: The timestamp of the data processing.
     - `version`: The version of the profile.

5. **Integration with OpenSearch**:
   - The script can send the generated JSON profiles to an OpenSearch instance for indexing.
   - It also supports retrieving existing profiles from OpenSearch to update or merge data.

## How to Use

1. **Setup**:
   - Ensure the required Python packages are installed. The script uses libraries such as `nltk`, `langdetect`, `gensim`, and `opensearch-py`.
   - Download the pre-trained Word2Vec model and update the `model_path` variable in the script.

2. **Run the Script**:
   - Place the raw data files in a folder (e.g., `path\people_search\people_profiles\`).
   - Execute the script:
     ```bash
     python generate_json.py
     ```

3. **Output**:
   - The script will process the data and print the normalized content.
   - Optionally, it can send the JSON profiles to OpenSearch for indexing.

## Requirements
- Python 3.7+
- Required Python packages:
  - `nltk`
  - `langdetect`
  - `gensim`
  - `opensearch-py`

## Notes
- Ensure the OpenSearch instance is configured and accessible.
- The script includes commented-out sections for additional functionality, such as using external APIs for keyword extraction or merging profiles with existing data.

## Future Improvements
- Enhance keyword extraction using external APIs.
- Add support for additional languages.
- Implement error handling and logging for better debugging.
