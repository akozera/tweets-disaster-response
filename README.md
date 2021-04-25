# Disaster Response Pipeline Project

This project aims to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) and to build a model for an API that classifies disaster messages.

The final product of this project is a web app where an emergency worker could input a new message and get classification results in several categories. It will also display some visualizations of the data.


### Deployment

To run this project, you will need to perfrom three steps:

1. Run the following commands in the project's root directory to set up the database and model.

    * Run the ETL pipeline that cleans data and stores in database:

    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

    * Run the ML pipeline that trains classifier and saves to a pickle file:

    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

2. Run the following command in the app's directory to run the web app:

    ```
    python run.py
    ```

3. Go to http://0.0.0.0:3001/.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

Credit to [Udacity](udacity.com) for providing materials based on which I could have built this repo.