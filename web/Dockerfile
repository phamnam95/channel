FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

RUN git clone https://github.com/ahay/src.git
WORKDIR /app/src
RUN ./configure API=python 
RUN make install
CMD source share/madagascar/etc/env.sh


WORKDIR /app
COPY static static/
COPY templates templates/

COPY main.py /app
CMD ["python", "main.py"]~

