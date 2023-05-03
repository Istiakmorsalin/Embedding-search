from dataloader import DataLoader
from embeddingsearch import EmbeddingSearch

def main():
    dataloader = DataLoader()
    embedding_search = EmbeddingSearch()
    img2vec_dict = dataloader.build_index()
    print(type(img2vec_dict))
    
if __name__ == '__main__':
    main()