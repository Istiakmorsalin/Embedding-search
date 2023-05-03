from redisoperations import RedisConnector
from deepface import DeepFace
from deepface.commons import functions, distance
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from embeddinggenerator import EmbeddingGenerator
from redis import Redis

class Utils: 
    def create_flat_index(redis_conn, vector_field_name, number_of_vectors, vector_dimensions=512, distance_metric='L2'):
        redis_conn.ft().create_index([
            VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }),
            TagField("product_type"),
            TextField("item_name"),
            TagField("country")        
        ])
        
    def create_hnsw_index(redis_conn, vector_field_name, number_of_vectors, vector_dimensions=512, distance_metric='L2', M=40, EF=200):
        redis_conn.ft().create_index([
            VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, "EF_CONSTRUCTION": EF}),
            TagField("product_type"),
            TextField("item_name"),
            TagField("country")     
        ]) 

    def load_vectors(client: Redis, product_metadata, vector_dict, vector_field_name):
        p = client.pipeline(transaction=False)

        for index in product_metadata.keys():    
            # hash key
            key = 'product:' + product_metadata[index]['primary_key']
        
            # hash values
            item_metadata = product_metadata[index]
            item_path = item_metadata['path']
        
            if item_path in vector_dict:
                # retrieve vector for product image 
                product_image_vector = vector_dict[item_path].astype(np.float32).tobytes()
                item_metadata[vector_field_name] = product_image_vector
            
                # HSET
                # p.hset(key, mapping=product_data_values)
                p.hset(key, mapping=item_metadata)
            
        p.execute()
