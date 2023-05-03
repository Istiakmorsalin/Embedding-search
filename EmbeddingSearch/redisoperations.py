from redis import Redis

class RedisConnector:

    def connect(self): 
        #Load Product data and truncate long text fields
        host = '127.0.0.1'
        port = 6379
        redis_conn = Redis(host = host, port = port)
        redis_conn.ping()
        print ('Connected to redis')

    # FLAT - Load and Index Product Data    