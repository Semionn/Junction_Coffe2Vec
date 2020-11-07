import sqlite3
import logging
import names
from numpy import random
import numpy as np
from operator import itemgetter
import datetime

logging.basicConfig(level=logging.DEBUG)


class CoffeeDB(object):
    WEATHER_DATE_IDX = 0
    WEATHER_TEMP_IDX = 1

    def __init__(self):
        self.db_path = "coffee_sqllite.db"

    def execute_query(self, query):
        with sqlite3.connect(self.db_path) as conn: # или :memory: чтобы сохранить в RAM
            cursor = conn.cursor()
            return cursor.execute(query)

    def drop_all_tables(self):
        drop_table_queries = self.execute_query("""
                select 'drop table ' || name || ';' from sqlite_master
                where type = 'table';
                """).fetchall()
        for query in drop_table_queries:
            self.execute_query(query[0])
        logging.info('All tables dropped')

    def rebuild_db(self):
        self.drop_all_tables()
        self.create_tables()
        self.insert_generated_data()

    def insert_generated_data(self):
        self.insert_cities()
        self.insert_customer_type()
        self.insert_product_type()
        self.insert_customers()
        self.insert_products()
        self.insert_weather()
        logging.info('Generated data inserted into the tables')

    def select_cities(self):
        return self.execute_query("""
            select *
            from city
            """).fetchall()

    def select_customer_types(self):
        return self.execute_query("""
            select ROWID, name
            from customer_type
            """).fetchall()

    def select_product_types(self):
        return self.execute_query("""
            select ROWID, name
            from product_type
            """).fetchall()

    def select_customers(self):
        return self.execute_query("""
            select *
            from customer_type
            """).fetchall()

    def select_weather(self):
        return self.execute_query("""
            select *
            from weather
            """).fetchall()

    def select_products(self):
        return self.execute_query("""
            select *
            from product
            """).fetchall()

    def get_start_date(self):
        return datetime.datetime(2018, 10, 1)

    def get_end_date(self):
        return datetime.datetime(2020, 11, 8)

    def insert_cities(self):
        cities = [
            'Helsinki',
            'Espoo',
            'Tampere',
            'Vantaa',
            'Oulu',
            'Turku',
            'Jyväskylä',
            'Lahti',
            'Kuopio',
            'Pori',
        ]
        for city in cities:
            self.execute_query(f"""INSERT INTO city VALUES
                              ('{city}')
                               """)

    def insert_customer_type(self):
        types = [
            'cafe',
            'restaurant',
            'hotel',
        ]
        for type in types:
            self.execute_query(f"""INSERT INTO customer_type VALUES
                              ('{type}')
                               """)

    def get_product_types(self):
        return {
            'coffee': ['samples', 'Tazza Low Lactose Hot Choco', 'Vendor Espresso', 'Paulig Special Medium beans', 'Tazza hot choco', 'Paulig Special Medium'],
            'coffeemachine': ['Thermoplan BLACK & WHITE', 'Nuova Simonelli Appia Life', 'Nuova Simonelli Aurelia Wave Digit', 'Jura E8 Platin Touch', 'Jura WE8'],
            'disposable': ['Vendor Whitener', 'paulig TA cup',  'Arla lact free portion milk', 'Paulig portionsugar brown'],
        }

    def insert_product_type(self):
        types = list(self.get_product_types().keys())
        for type in types:
            self.execute_query(f"""INSERT INTO product_type VALUES
                              ('{type}')
                               """)

    def insert_customers(self, count: int = 100):
        customer_types = list(map(itemgetter(0), self.select_customer_types()))
        customer_types_count = len(customer_types)

        probabilities = self.make_probabilities(customer_types_count, [0.6, 0.3])

        for i in range(count):
            name = names.get_full_name()
            customer_type = random.choice(customer_types, p=probabilities)
            self.execute_query(f"""INSERT INTO customer VALUES
                              ('{name}', '{customer_type}')
                               """)
        # for customer in self.select_customers():
        #     logging.info(customer)

    def make_probabilities(self, choices_count, default_probabilities=None):
        probabilities = [1 / choices_count] * choices_count
        rest = 1
        default_count = 0
        if default_probabilities is not None:
            default_count = len(default_probabilities)
            for i, prob in enumerate(default_probabilities):
                probabilities[i] = prob
                rest -= prob
        for i in range(default_count, choices_count):
            probabilities[i] = rest / (choices_count - default_count)
        assert np.abs(sum(probabilities) - 1) < 1e-5
        return probabilities

    def insert_products(self, count: int = 200):
        product_types = list(map(itemgetter(0), self.select_product_types()))
        product_types_names = list(map(itemgetter(1), self.select_product_types()))
        product_types_count = len(product_types)

        type_probabilities = self.make_probabilities(product_types_count, [0.55, 0.4])

        product_names = self.get_product_types()

        costs = np.exp(random.poisson(5, count))
        for i in range(count):
            product_type_index = random.choice(range(len(product_types)), p=type_probabilities)
            product_type = product_types[product_type_index]
            product_type_names = product_names[product_types_names[product_type_index]]
            product_probabilities = self.make_probabilities(len(product_type_names))
            product_name = random.choice(product_type_names, p=product_probabilities)
            cost = np.round(costs[i], 2)
            self.execute_query(f"""INSERT INTO product VALUES
                              ('{product_name}', '{cost}', '{product_type}')
                               """)
        # for product in self.select_products():
        #     logging.info(product)

    def generate_temperature(self, date: datetime.datetime):
        min_temp = -10
        max_temp = 30

        day_of_year = date.timetuple().tm_yday
        shifts_count = 20
        local_shift = np.cos(day_of_year * np.pi * shifts_count / 365) * 2

        hottest_day = 365/2
        max_distance = hottest_day
        hottest_day_distance = abs(day_of_year - hottest_day)
        temp_range = (max_temp - min_temp)
        x = hottest_day_distance / max_distance
        result = min_temp + temp_range * (np.cos(x * np.pi) + 1) / 2 + random.normal(0, temp_range * 0.05) + local_shift
        if date.hour < 9 or date.hour > 21:
            result -= 3  # night temp rough bias
        return result

    def insert_weather(self, count_per_day=2):
        cur_date = self.get_start_date()
        end_date = self.get_end_date()
        if count_per_day > 8:
            count_per_day = 8

        while cur_date < end_date:
            date_time = cur_date
            for i in range(count_per_day):
                date_time = date_time + datetime.timedelta(hours=random.randint(3, int(24 / count_per_day)))
                temperature = self.generate_temperature(date_time)
                wind_speed = abs(random.normal(10, 2))
                pressure = abs(random.normal(760, 2))
                humidity_logit = random.normal(2, 1)
                humidity = 1 / (1 + np.exp(-humidity_logit))
                precip_prob = 1 / (1 + np.exp(-(humidity_logit - 0.2)))
                self.execute_query(f"""INSERT INTO weather VALUES
                                  ('{date_time}', '{temperature}','{wind_speed}', '{pressure}', '{precip_prob}', '{humidity}')
                                   """)
            cur_date += datetime.timedelta(days=1)
        logging.info(cur_date)
        # for weather in self.select_weather()[-300:]:
        #     logging.info(weather)

    def create_tables(self):
        self.execute_query("""CREATE TABLE city
                          (name text)
                           """)
        self.execute_query("""CREATE TABLE customer_type
                          (name text)
                           """)
        self.execute_query("""CREATE TABLE customer
                          (name text,
                          type integer,
                           FOREIGN KEY(type) REFERENCES customer_type(ROWID))
                           """)
        self.execute_query("""CREATE TABLE cafe
                          (name text,
                          address text,
                          customer_id integer,
                          city_id integer,
                           FOREIGN KEY(customer_id) REFERENCES customer(ROWID),
                           FOREIGN KEY(city_id) REFERENCES city(ROWID))
                           """)
        self.execute_query("""CREATE TABLE product_type
                          (name text)
                           """)
        self.execute_query("""CREATE TABLE product
                          (name text,
                          cost integer,
                          type integer,
                           FOREIGN KEY(type) REFERENCES product_type(ROWID))
                           """)
        self.execute_query("""CREATE TABLE weather
                          (date_time datetime,
                          temp integer,
                          wind_speed integer,
                          pressure integer,
                          precip_prob integer,
                          humidity integer)
                           """)
        # price per one piece
        # revenue = quantity * (price - cost)
        # denormalized scheme to avoid extra joins
        self.execute_query("""CREATE TABLE orders
                          (date_time datetime,
                          quantity integer,
                          revenue integer,
                          price integer, 
                          product_id integer,
                          product_type_id integer,
                          customer_id integer,
                          city_id integer,
                          cafe_id integer,
                          weather_id integer,
                           FOREIGN KEY(product_id) REFERENCES product(ROWID),
                           FOREIGN KEY(product_type_id) REFERENCES product_type(ROWID),
                           FOREIGN KEY(customer_id) REFERENCES customer(ROWID),
                           FOREIGN KEY(city_id) REFERENCES city(ROWID),
                           FOREIGN KEY(cafe_id) REFERENCES cafe(ROWID),
                           FOREIGN KEY(weather_id) REFERENCES weather(ROWID)
                           )
                           """)
        logging.info('Tables recreated')
