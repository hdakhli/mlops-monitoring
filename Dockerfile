FROM python:3.10

WORKDIR .

COPY requirements.txt ./
COPY titanic_model.joblib ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./api_titanic.py" ]
