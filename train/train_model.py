import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def ingest_data(file_path: str) -> pd.DataFrame:
    # Load data from file
    titanic = pd.read_excel(file_path)
    return titanic


def preprocess_data(titanic: pd.DataFrame) -> pd.DataFrame:
    # Drop columns
    titanic = titanic[['survived', 'pclass', 'sex', 'age']]
    # Fill missing values
    titanic.dropna(axis=0, inplace=True)
    # Convert categorical to numerical
    titanic['sex'] = titanic['sex'].map(lambda x: 1 if x == 'male' else 0)
    return titanic


def train_model(titanic: pd.DataFrame):
    # Train model
    model = KNeighborsClassifier(n_neighbors=3)

    y = titanic['survived']
    x = titanic.drop('survived', axis=1)

    model.fit(x, y)
    model.score(x, y)

    return model


def save_model(model, file_path: str):
    # Save model
    joblib.dump(model, file_path)
    return file_path


def main():
    # Ingest data
    titanic = ingest_data('train/data/titanic.xls')

    # Preprocess data
    preprocessed = preprocess_data(titanic)

    # Train model
    model = train_model(preprocessed)

    # Save model
    save_model(model, 'titanic_model.joblib')


if __name__ == "__main__":
    main()
