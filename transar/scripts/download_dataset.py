import os

# API credentials
os.environ['KAGGLE_USERNAME'] = 'dicency'
os.environ['KAGGLE_KEY'] = '18cf1320dbb5d1f5a5906950c04cddbc'

import kaggle

def download_dataset():
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    #dataset = 'ayuraj/asl-dataset' 
    dataset = 'grassknoted/asl-alphabet'


    # Download the dataset
    kaggle.api.dataset_download_files(dataset, path=dataset_path, unzip=True)

    print(f'Dataset downloaded and extracted to {dataset_path}')

if __name__ == '__main__':
    download_dataset()
