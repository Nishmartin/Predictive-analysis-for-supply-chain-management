import XGBoost
import DecisionTree
import GradientBoosting
import RandomForest
import NeuralNetwork

def run_all_models():
    print("Running XGBoost model:")
    XGBoost.train_model(file_path)
    print("\nRunning Decision Tree model:")
    DecisionTree.train_model(file_path)
    print("\nRunning Gradient Boosting model:")
    GradientBoosting.train_model(file_path)
    print("\nRunning Random Forest model:")
    RandomForest.train_model(file_path)
    print("\nRunning Neural Network model:")
    NeuralNetwork.train_model(file_path)

if __name__ == "__main__":
    file_path = r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv"
    run_all_models()
