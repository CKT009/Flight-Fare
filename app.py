from flask import Flask, request, render_template, jsonify
from src.pipeline.train_pipeline import TrainPipeline  # Adjust import path as needed
from src.pipeline.prediction_pipeline import PredictionPipeline  # Adjust import path as needed
from src.exception import customexception
import sys
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Home Page"})

@app.route('/train')
def start_training():
    logging.info("Received POST request at /train")
    try:
        # Instantiate the pipeline
        pipeline = TrainPipeline()

        # Run the training pipeline
        pipeline.run_pipeline()

        return jsonify({"message": "Training pipeline executed successfully."}), 200

    except customexception as e:
        logging.error(f"CustomException occurred: {e}")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred. Please check the logs for details."}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load the model and preprocessor
            model_path = "artifacts/best_model.pkl"  # Adjust path as needed
            preprocessor_path = "artifacts/preprocessor.pkl"  # Adjust path as needed
            prediction_pipeline = PredictionPipeline(model_path, preprocessor_path)

            # Get input data from form
            input_data = {
                "airline": request.form.get("airline"),
                "date_of_journey": request.form.get("date_of_journey"),
                "source": request.form.get("source"),
                "destination": request.form.get("destination"),
                "duration": float(request.form.get("duration")),
                "total_stops": request.form.get("total_stops")
            }

            # Perform prediction
            result = prediction_pipeline.predict(input_data)

            return jsonify({"prediction": result.tolist()}), 200

        except customexception as e:
            logging.error(f"CustomException occurred: {e}")
            return jsonify({"error": str(e)}), 500

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return jsonify({"error": "An unexpected error occurred. Please check the logs for details."}), 500
    else:
        return render_template("predict.html")


if __name__ == '__main__':
    # Set host and port if needed, e.g., host='0.0.0.0', port=5000
    app.run(debug=True)
