# Skin Care Recommendation System

A data science project that provides personalized skin care product recommendations based on user characteristics, product reviews, and ratings.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

This project implements a comprehensive skin care recommendation system using machine learning and natural language processing techniques. The system analyzes user characteristics, product reviews, and ratings to provide personalized skin care product recommendations.

The recommendation engine combines:
- User-based filtering (using skin type, tone, and other characteristics)
- Content-based filtering (analyzing product reviews and ingredients)
- Collaborative filtering elements (using ratings and reviews from similar users)

## Features

- **Personalized Recommendations**: Provides product recommendations based on individual user characteristics
- **Text Analysis**: Uses NLP to analyze product reviews and extract meaningful insights
- **Multiple Recommendation Approaches**: Implements both user-based and content-based recommendation algorithms
- **Data Visualization**: Includes visualizations for user demographics and product analysis
- **Product Similarity**: Finds similar products based on ingredients and reviews using cosine similarity

## Dataset

The system uses a comprehensive skin care dataset containing:
- User information: Username, Skin_Tone, Skin_Type, Eye_Color, Hair_Color
- Product information: Product name, Brand, Price
- User feedback: Rating_Stars, Review text
- Additional derived features for analysis

## Technologies Used

- **Programming Language**: Python 3
- **Data Analysis**: pandas, NumPy
- **Machine Learning**: scikit-learn (RandomForest, Naive Bayes, SGDClassifier)
- **Natural Language Processing**: NLTK (tokenization, stemming, lemmatization)
- **Text Vectorization**: CountVectorizer, TfidfVectorizer
- **Data Visualization**: matplotlib, bokeh
- **Text Visualization**: wordcloud
- **Web Framework**: ipywidgets (for interactive components)
- **Development Environment**: Jupyter Notebook

## Methodology

The project follows a standard data science workflow:

1. **Data Loading and Exploration**
   - Load skin care dataset using pandas
   - Initial data exploration and understanding

2. **Exploratory Data Analysis**
   - Visualize user demographics (skin tone, skin type, eye color, hair color)
   - Analyze product distributions and ratings

3. **Data Preprocessing**
   - Clean text data in reviews
   - Handle missing values
   - Prepare data for machine learning models

4. **Feature Engineering**
   - Convert text reviews into numerical features using NLP techniques
   - Create TF-IDF vectors for text analysis
   - Generate word clouds for text visualization

5. **Model Development**
   - Implement multiple machine learning algorithms
   - Train models to predict user preferences
   - Evaluate model performance

6. **Recommendation Generation**
   - Generate personalized product recommendations
   - Implement content-based and user-based recommendation approaches
   - Calculate product similarities using cosine similarity

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skin-care-recommendation-system.git
   ```

2. Navigate to the project directory:
   ```bash
   cd skin-care-recommendation-system
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (if not already installed):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Skin Care Recommendation Sytem.ipynb"
   ```

2. Run the cells sequentially to:
   - Load and explore the data
   - Perform exploratory data analysis
   - Preprocess the data
   - Train machine learning models
   - Generate recommendations

3. To get recommendations based on user features:
   ```python
   # Enter your characteristics when prompted
   skintone = input("Enter Skin Tone: ")
   eyecolor = input("Enter Eye Color: ")
   skintype = input("Enter Skin Type: ")
   haircolor = input("Enter Hair Color: ")
   
   # Get personalized recommendations
   recommend_products_by_user_features(skintone, eyecolor, skintype, haircolor)
   ```

4. To get recommendations based on product similarity:
   ```python
   # Get similar products
   content_recommendation('Product Name')
   ```

## Results

The system successfully provides personalized skin care recommendations using multiple approaches:
- User-based recommendations based on individual characteristics
- Content-based recommendations based on product similarity
- Machine learning models to predict user preferences

The recommendation engine helps users discover skin care products that are more likely to work for their specific skin type and preferences.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to the skin care community for providing the dataset
- Inspired by recommendation systems in e-commerce and beauty industries