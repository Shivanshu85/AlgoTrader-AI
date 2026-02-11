import os
import logging
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class KaggleDataIngestion:
    """
    Ingest data from Kaggle datasets using kagglehub.
    """
    
    def __init__(self, dataset_handle: str = "mrsimple07/stock-price-prediction"):
        """
        Initialize the Kaggle ingestion client.
        
        Args:
            dataset_handle: The Kaggle dataset handle (username/dataset-slug)
        """
        self.dataset_handle = dataset_handle
        
        # Ensure credentials exist
        if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) and not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            logger.warning("Kaggle credentials not found. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set.")

    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Load a specific ticker file from the dataset.
        
        Args:
            ticker: The stock ticker symbol (e.g., "AAPL")
            
        Returns:
            pd.DataFrame: The loaded data
        """
        # The dataset likely contains files named "{Ticker}.csv"
        file_path = f"{ticker}.csv"
        
        logger.info(f"Downloading {file_path} from {self.dataset_handle}...")
        
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            self.dataset_handle,
            file_path
        )
        
        logger.info(f"Successfully loaded {len(df)} records for {ticker}")
        return df