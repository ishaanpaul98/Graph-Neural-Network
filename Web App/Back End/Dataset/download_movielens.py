import os
import requests
import zipfile
import pandas as pd
from pathlib import Path

class MovieLensDownloader:
    """
    Class to download and process MovieLens datasets
    """
    def __init__(self, dataset_size='100k'):
        """
        Initialize the downloader with dataset size
        Args:
            dataset_size (str): Size of the dataset ('100k', '1m', '10m', '20m', '32m')
        """
        self.dataset_size = dataset_size.lower()
        self.base_url = "https://files.grouplens.org/datasets/movielens"
        self.dataset_folder = Path("dataset")
        #self.dataset_folder.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and names
        self.dataset_info = {
            '100k': {
                'url': f"{self.base_url}/ml-100k.zip",
                'name': 'ml-100k'
            },
            '1m': {
                'url': f"{self.base_url}/ml-1m.zip",
                'name': 'ml-1m'
            },
            '10m': {
                'url': f"{self.base_url}/ml-10m.zip",
                'name': 'ml-10m'
            },
            '20m': {
                'url': f"{self.base_url}/ml-20m.zip",
                'name': 'ml-20m'
            },
            '32m': {
                'url': f"{self.base_url}/ml-25m.zip",  # Note: 32M is actually 25M
                'name': 'ml-25m'
            }
        }
        
        if self.dataset_size not in self.dataset_info:
            raise ValueError(f"Invalid dataset size. Choose from: {', '.join(self.dataset_info.keys())}")
    
    def download(self):
        """
        Download the specified MovieLens dataset
        """
        dataset_info = self.dataset_info[self.dataset_size]
        zip_path = self.dataset_folder / f"{dataset_info['name']}.zip"
        
        # Download if file doesn't exist
        if not zip_path.exists():
            print(f"Downloading MovieLens {self.dataset_size} dataset...")
            response = requests.get(dataset_info['url'], stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed!")
        
        # Extract if not already extracted
        extract_path = self.dataset_folder / dataset_info['name']
        if not extract_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_folder)
            print("Extraction completed!")
        
        return extract_path
    
    def get_dataset_info(self):
        """
        Get information about the downloaded dataset
        """
        dataset_path = self.dataset_folder / self.dataset_info[self.dataset_size]['name']
        
        if not dataset_path.exists():
            print("Dataset not found. Please download it first.")
            return None
        
        try:
            # For ml-100k dataset
            if self.dataset_size == '100k':
                # Read ratings from u.data
                ratings_file = dataset_path / 'u.data'
                if not ratings_file.exists():
                    raise FileNotFoundError(f"No ratings file found at {ratings_file}")
                
                ratings_df = pd.read_csv(ratings_file, sep='\t', header=None,
                                       names=['userId', 'movieId', 'rating', 'timestamp'])
                
                # Read movies from u.item
                movies_file = dataset_path / 'u.item'
                if not movies_file.exists():
                    raise FileNotFoundError(f"No movies file found at {movies_file}")
                
                movies_df = pd.read_csv(movies_file, sep='|', header=None, encoding='latin-1',
                                      names=['movieId', 'title', 'release_date', 'video_release_date',
                                            'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                            'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
            
            # For other datasets
            else:
                # Find ratings file (could be .dat or .csv)
                ratings_files = list(dataset_path.glob('ratings.*'))
                if not ratings_files:
                    raise FileNotFoundError(f"No ratings file found in {dataset_path}")
                ratings_file = ratings_files[0]
                
                # Find movies file (could be .dat or .csv)
                movies_files = list(dataset_path.glob('movies.*'))
                if not movies_files:
                    raise FileNotFoundError(f"No movies file found in {dataset_path}")
                movies_file = movies_files[0]
                
                # Read the files based on their extension
                if ratings_file.suffix == '.dat':
                    ratings_df = pd.read_csv(ratings_file, sep='::', engine='python', 
                                           names=['userId', 'movieId', 'rating', 'timestamp'])
                else:
                    ratings_df = pd.read_csv(ratings_file)
                    
                if movies_file.suffix == '.dat':
                    movies_df = pd.read_csv(movies_file, sep='::', engine='python', 
                                          names=['movieId', 'title', 'genres'])
                else:
                    movies_df = pd.read_csv(movies_file)
            
            print("\nDataset Information:")
            print(f"Total number of ratings: {len(ratings_df)}")
            print(f"Total number of movies: {len(movies_df)}")
            print(f"Total number of users: {ratings_df['userId'].nunique()}")
            print(f"Rating range: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
            
            return {
                'ratings': ratings_df,
                'movies': movies_df
            }
            
        except Exception as e:
            print(f"Error reading dataset files: {str(e)}")
            return None

def main():
    # Example usage
    sizes = ['100k', '1m', '10m', '20m', '32m']
    print("Available MovieLens dataset sizes:", ', '.join(sizes))
    
    while True:
        size = input("\nEnter the dataset size to download (or 'q' to quit): ").lower()
        if size == 'q':
            break
            
        try:
            downloader = MovieLensDownloader(size)
            dataset_path = downloader.download()
            print(f"\nDataset downloaded and extracted to: {dataset_path}")
            downloader.get_dataset_info()
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 