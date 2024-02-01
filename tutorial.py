from main import *

recommender = Recommender()
recommendations = recommender.model_recommendations(
                                                [86320], #id просмотренных фильмов
                                                [5], #оценки фильмов
                                                N=10 
                                                )
print(recommendations) #id рекоммендаций


# вот так, если передаёшь imdb id:
imdb_map = load_imdb_map()
recommendations = recommender.model_recommendations(
                                                imdb_map.from_imdb([1527186]), #imdb id просмотренных фильмов
                                                [5], #оценки фильмов
                                                N=10 
                                                )
print(load_imdb_map().to_imdb(recommendations)) #imdb id

