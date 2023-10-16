import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd
from prometheus_client import Counter, make_asgi_app

survived_counter = Counter("survived", "Counter for survived")
not_survived_counter = Counter("not_survived", "Counter for not survived")

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
titanic_model = joblib.load("./titanic_model.joblib")


@app.get("/titanic")
def prediction_api(pclass: int, sex: int, age: int):
    x = [pclass, sex, age]
    prediction = titanic_model.predict(pd.DataFrame(x).transpose())
    survived = int(prediction) == 1
    if survived:
        survived_counter.inc()
    else:
        not_survived_counter.inc()
    return survived


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
