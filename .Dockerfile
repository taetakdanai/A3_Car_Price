FROM python:3

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install joblib
RUN pip3 install scikit-learn

COPY . /root/code

CMD tail -f /dev/null