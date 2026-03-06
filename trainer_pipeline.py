from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def run_pipeline(self):

        # Step 1 : Data Ingestion
        ingestion = DataIngestion()
        data = ingestion.Ingest_data(r"C:\Users\rohit\OneDrive\Desktop\loan approval project\data\train_u6lujuX_CVtuZ9i.csv")

        print("Data Ingestion Completed")

        # Step 2 : Data Preprocessing
        preprocessing = DataPreprocessing()
        X, y, preprocessor = preprocessing.Preprocessdata(data)

        print("Data Preprocessing Completed")

        # Step 3 : Model Training
        trainer = ModelTrainer()
        trainer.TrainModel(X, y, preprocessor)

        print("Model Training Completed")


if __name__ == "__main__":

    pipeline = TrainPipeline()
    pipeline.run_pipeline()