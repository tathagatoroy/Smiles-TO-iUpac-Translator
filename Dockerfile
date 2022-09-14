FROM python:3.9.12

COPY Python_requirements.txt .

RUN pip install -r Python_requirements.txt

WORKDIR /STOUT

ADD predictor_demo.py .

CMD ["python", "./predictor_demo.py"]