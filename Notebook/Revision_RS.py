"""

- The objective:
    recommend relevant items for users, based on their preference

- The main families of methods for RecSys are: 

    1- Content-Based Filtering:
        This method uses only information about the description and attributes of the items users has previously consumed to model user's preferences.
        In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present).
        In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended.  


    2- Collaborative Filtering:
        This method makes automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating).
        The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on a set of items, A is more likely to have B's opinion for a given item than that of a randomly chosen person.                

        

    3- Hybrid methods:  
        Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective than pure approaches in some cases.
        These methods can also be used to overcome some of the common problems in recommender systems such as cold start and the sparsity problem.    
"""


# Deskdrop dataset which contains a real sample of 12 months logs (Mar. 2016 - Feb. 2017) from CI&T's Internal Communication platform (DeskDrop).
#  It contains about 73k logged users interactions on more than 3k public articles shared in the platform.

# It is composed of two CSV files:  
# - **shared_articles.csv**
# - **users_interactions.csv**


# shared_articles.csv: 
    # Contains information about the articles shared in the platform.
    # Each article has its sharing date (timestamp), the original url, title,
    # content in plain text, the article' lang (Portuguese: pt or English: en) and information about the user who shared the article (author).           

#     There are two possible event types at a given timestamp: 
#         - CONTENT SHARED: The article was shared in the platform and is available for users. 
#         - CONTENT REMOVED: The article was removed from the platform and not available for further recommendation.

    # For the sake of simplicity, we only consider here the "CONTENT SHARED" event type,
    #  assuming  that all articles were available during the whole one year period.
    #  For a more precise evaluation (and higher accuracy), only articles that were available at a given time should be recommended, but we let this exercice for you.


# users_interactions.csv:
#   Contains logs of user interactions on shared articles. It can be joined to **articles_shared.csv** by **contentId** column.

#    The eventType values are:  
#       - **VIEW**: The user has opened the article. 
#       - **LIKE**: The user has liked the article. 
#       - **COMMENT CREATED**: The user created a comment in the article. 
#       - **FOLLOW**: The user chose to be notified on any new comment in the article. 
#       - **BOOKMARK**: The user has bookmarked the article for easy return in the future.    


# Data munging:
    # As there are different interactions types, we associate them with a weight or strength, assuming that,
    #  for example, a comment in an article indicates a higher interest of the user on the item than a like, or than a simple view.


# Recommender systems have a problem known as [user cold-start]:
    # in which is hard do provide personalized recommendations for users with none or a very few number of consumed items, due to the lack of information to model their preferences.  
    # For this reason, we are keeping in the dataset only users with at leas 5 interactions.    


# In Deskdrop, users are allowed to view an article many times, and interact with them in different ways (eg. like or comment).
#  Thus, to model the user interest on a given article, we aggregate all the interactions the user has performed in an item by a weighted sum of interaction type strength and apply a log transformation to smooth the distribution.    


# Evaluation:
    # Evaluation is important for machine learning projects, because it allows to compare objectivelly different algorithms and hyperparameter choices for models.  
    # One key aspect of evaluation is to ensure that the trained model generalizes for data it was not trained on, using **Cross-validation** techniques.
    #  We are using here a simple cross-validation approach named **holdout**, in which a random data sample (20% in this case) are kept aside in the training process, 
    # and exclusively used for evaluation. All evaluation metrics reported here are computed using the **test set**.



# Popularity model:
    # A common (and usually hard-to-beat) baseline approach is the Popularity model.
    #  This model is not actually personalized - it simply recommends to a user the most popular items that the user has not previously consumed.
    #  As the popularity accounts for the "wisdom of the crowds", it usually provides good recommendations, generally interesting for most people.   
    # The main objective of a recommender system is to leverage the long-tail items to the users with very specific interests, which goes far beyond this simple technique.




# Content-Based Filtering model:
    # Content-based filtering approaches leverage description or attributes from items the user has interacted to recommend similar items.
    #  It depends only on the user previous choices, making this method robust to avoid the *cold-start* problem.
    # For textual items, like articles, news and books, it is simple to use the raw text to build item profiles and user profiles.  
    # Here we are using a very popular technique in information retrieval (search engines) named [TF-IDF]



# To model the user profile, we take all the item profiles the user has interacted and average them.
#  The average is weighted by the interaction strength, in other words, the articles the user has interacted the most (eg. liked or commented) will have a higher strength in the final user profile.   




# Collaborative Filtering model:
    # Collaborative Filtering (CF) has two main implementation strategies:  
        # Memory-based: 
            # This approach uses the memory of previous users interactions to compute users similarities based on items they've interacted (user-based approach) or compute items similarities based on the users that have interacted with them (item-based approach).  

            # A typical example of this approach is User Neighbourhood-based CF, in which the top-N similar users (usually computed using Pearson correlation) for a user are selected and used to recommend items those similar users liked,
            # but the current user have not interacted yet. This approach is very simple to implement, but usually do not scale well for many users. 

        # Model-based:
        #  This approach, models are developed using different machine learning algorithms to recommend items to users. There are many model-based CF algorithms,
        #  like neural networks, bayesian networks, clustering models, and latent factor models such as Singular Value Decomposition (SVD) and, probabilistic latent semantic analysis.




# Matrix Factorization:
    # Latent factor models compress user-item matrix into a low-dimensional representation in terms of latent factors. One advantage of using this approach is that instead of having a high dimensional matrix containing abundant number of missing values we will be dealing with a much smaller matrix in lower-dimensional space.  
    # A reduced There are several advantages with this paradigm. It handles the sparsity of the original matrix better than memory based ones. Also comparing similarity on the resulting matrix is much more scalable especially in dealing with large sparse datasets.  


# An important decision is the number of factors to factor the user-item matrix.
#  The higher the number of factors, the more precise is the factorization in the original matrix reconstructions.
#  Therefore, if the model is allowed to  memorize too much details of the original matrix, it may not generalize well for data it was not trained on.
#  Reducing the number of factors increases the model generalization.



# After the factorization, we try to to reconstruct the original matrix by multiplying its factors. 
# The resulting matrix is not sparse any more. It was generated predictions for items the user have not yet interaction, which we will exploit for recommendations.



#  Evaluating the Collaborative Filtering model (SVD matrix factorization),
#  we observe that we got **Recall@5 (33%)** and **Recall@10 (46%)** values, much higher than Popularity model and Content-Based model.



# Hybrid Recommender:
    # What if we combine Collaborative Filtering and Content-Based Filtering approaches?    
    # Would that provide us with more accurate recommendations?    
    # In fact, hybrid methods have performed better than individual approaches in many studies and have being extensively used by researchers and practioners.  
    # Let's build a simple hybridization method, as an ensemble that takes the weighted average of the normalized CF scores with the Content-Based scores, and ranking by resulting score. In this case, as the CF model is much more accurate than the CB model, the weights for the CF and CB models are 100.0 and 1.0, respectivelly.


# We have a new champion:  
    # Our simple hybrid approach surpasses Content-Based filtering with its combination with Collaborative Filtering. 
    # Now we have a **Recall@5** of **34.2%** and **Recall@10** of **47.9%**    


# Comparing the methods    

# Testing:
    # Let's test the best model (Hybrid) for my user.

#  The recommendations really matches my interests, as I would read all of them 

# Conclusion:
    # In this notebook, we've explored and compared the main Recommender Systems techniques on [CI&T Deskdrop]dataset. 
    # It could be observed that for articles recommendation, content-based filtering and a hybrid method performed better than Collaborative Filtering alone.  
