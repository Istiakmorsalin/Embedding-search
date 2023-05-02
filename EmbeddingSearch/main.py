import random
import numpy as np
import pandas as pd
import time
from dataloader import DataLoader
from embeddinggenerator import EmbeddingGenerator

def main():
    dataloader = DataLoader()
    dataloader.loadData()
    embeddinggenerator = EmbeddingGenerator()

if __name__ == '__main__':
    main()