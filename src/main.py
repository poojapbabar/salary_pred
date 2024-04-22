from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.prediction_pipeline import PredictPipeline

if __name__ == "__main__":
    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    transformed_data, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_path)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_data, test_data, preprocessor_path)

    # Step 4: Prediction
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(features)

    print("Prediction:", prediction)
