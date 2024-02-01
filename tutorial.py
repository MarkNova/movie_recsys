from main import *

recommender = Recommender()
recommendations = recommender.model_recommendations(
                                                [86320], #imdb id просмотренных фильмов
                                                [5], #оценки фильмов
                                                N=10 #количество рекоммендаций
                                                )
print(recommendations) #внутренние id

print(load_imdb_map().to_imdb(recommendations)) #imdb id

