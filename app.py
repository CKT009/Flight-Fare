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
    return render_template("home.html")

@app.route('/train')
def start_training():
    logging.info("Received request at /train")
    try:
        # Instantiate the pipeline
        pipeline = TrainPipeline()

        # Run the training pipeline and get the model scores and best model info
        model_scores, best_model_name = pipeline.run_pipeline()

        return render_template("train.html", model_scores=model_scores, best_model_name=best_model_name)

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
                "Airline": request.form.get("airline"),
                "date_of_journey": request.form.get("date_of_journey"),
                "Source": request.form.get("source"),
                "Destination": request.form.get("destination"),
                "Duration": float(request.form.get("duration")),
                "Total_Stops": request.form.get("total_stops")
            }

            # Perform prediction
            result = prediction_pipeline.predict(input_data)

            # Pass the prediction result to the template
            return render_template("predict.html", prediction=result.tolist()[0])

        except customexception as e:
            logging.error(f"CustomException occurred: {e}")
            return render_template("predict.html", error=str(e))

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return render_template("predict.html", error="An unexpected error occurred. Please check the logs for details.")
    else:
        return render_template("predict.html")



if __name__ == '__main__':
    app.run(debug=False)
