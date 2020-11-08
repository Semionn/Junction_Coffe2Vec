import sqlite3
import logging
import names
from numpy import random
import numpy as np
from operator import itemgetter
import datetime
import pandas as pd
import tqdm

logging.basicConfig(level=logging.DEBUG)


class CoffeeDB(object):
    WEATHER_IDX = 0
    WEATHER_DATE = 1
    WEATHER_TEMP = 2
    WEATHER_WIND_SPEED = 3
    WEATHER_PRESSURE = 4
    WEATHER_PRECIP_PROB = 5
    WEATHER_HUMIDITY = 6

    PRODUCT_IDX = 0
    PRODUCT_NAME = 1
    PRODUCT_TYPE = 3
    PRODUCT_COST = 3

    CUSTOMER_IDX = 0
    CUSTOMER_NAME = 1
    CUSTOMER_TYPE = 2

    CITY_IDX = 0
    CITY_NAME = 1

    CAFE_IDX = 0
    CAFE_NAME = 1

    ORDER_INDEX = 0
    ORDER_DATE_TIME = 1
    ORDER_QUANTITY = 2
    ORDER_REVENUE = 3
    ORDER_PRICE = 4
    ORDER_PRODUCT_ID = 5
    ORDER_PRODUCT_TYPE_ID = 6
    ORDER_CUSTOMER_ID = 7
    ORDER_CITY_ID = 8
    ORDER_CAFE_ID = 9
    ORDER_WEATHER_ID = 10

    def __init__(self):
        self.db_path = "coffee_sqllite.db"
        self.conn = None

    def init_connection(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None
        self.conn = sqlite3.connect(self.db_path)

    def close_connection(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def execute_query(self, query):
        if self.conn is None:
            self.init_connection()
        cursor = self.conn.cursor()
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

    def rebuild_orders(self):
        self.execute_query("drop table orders;")
        self.create_orders_table()
        self.insert_orders()

    def insert_generated_data(self):
        self.insert_cities()
        self.insert_customer_type()
        self.insert_product_type()
        self.insert_customers()
        self.insert_products()
        self.insert_cafe()
        self.insert_weather()
        self.insert_orders()
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

    def select_products(self):
        return self.execute_query("""
            select ROWID, type, name, cost
            from product
            """).fetchall()

    def select_city_df(self):
        query_result = self.execute_query("""
            select ROWID, name
            from city
            """).fetchall()
        indices = list(map(itemgetter(self.CITY_IDX), query_result))
        names = list(map(itemgetter(self.CITY_NAME), query_result))
        return pd.DataFrame({'idx': indices, 'name': names})

    def select_orders_df(self):
        query_result = self.execute_query("""
            select ROWID, date_time, quantity, revenue, price, product_id, product_type_id, customer_id, city_id, cafe_id, weather_id
            from orders
            """).fetchall()
        result = {}
        names = {
            'idx': self.ORDER_INDEX,
            'dt': self.ORDER_DATE_TIME,
            'quantity': self.ORDER_QUANTITY,
            'revenue': self.ORDER_REVENUE,
            'price': self.ORDER_PRICE,
            'product_id': self.ORDER_PRODUCT_ID,
            'product_type_id': self.ORDER_PRODUCT_TYPE_ID,
            'customer_id': self.ORDER_CUSTOMER_ID,
            'city_id': self.ORDER_CITY_ID,
            'cafe_id': self.ORDER_CAFE_ID,
            'weather_id': self.ORDER_WEATHER_ID,
        }
        for name, field in names.items():
            result[name] = list(map(itemgetter(field), query_result))
        return pd.DataFrame(result)

    def select_cafe_df(self):
        query_result = self.execute_query("""
            select ROWID, name
            from cafe
            """).fetchall()
        indices = list(map(itemgetter(self.CAFE_IDX), query_result))
        names = list(map(itemgetter(self.CAFE_NAME), query_result))
        return pd.DataFrame({'idx': indices, 'name': names})

    def select_customers_df(self):
        query_result = self.execute_query("""
            select ROWID, name, type
            from customer
            """).fetchall()
        indices = list(map(itemgetter(self.CUSTOMER_IDX), query_result))
        names = list(map(itemgetter(self.CUSTOMER_NAME), query_result))
        types = list(map(itemgetter(self.CUSTOMER_TYPE), query_result))
        return pd.DataFrame({'idx': indices, 'name': names, 'type': types})

    def select_weather(self):
        return self.execute_query("""
            select *
            from weather
            """).fetchall()

    def select_weather_df(self):
        query_result = self.execute_query("""
            select ROWID, date_time, temp, wind_speed, pressure, precip_prob, humidity
            from weather
            """).fetchall()
        result = {}

        names = {
            'idx': self.WEATHER_IDX,
            'dt': self.WEATHER_DATE,
            'temp': self.WEATHER_TEMP,
            'wind_speed': self.WEATHER_WIND_SPEED,
            'pressure': self.WEATHER_PRESSURE,
            'precip_prob': self.WEATHER_PRECIP_PROB,
            'humidity': self.WEATHER_HUMIDITY,
        }
        for name, field in names.items():
            result[name] = list(map(itemgetter(field), query_result))
        return pd.DataFrame(result)

    def select_products_df(self):
        query_result = self.execute_query("""
            select ROWID, name, type, cost
            from product
            """).fetchall()
        result = {}
        names = {
            'idx': self.PRODUCT_IDX,
            'name': self.PRODUCT_NAME,
            'type': self.PRODUCT_TYPE,
            'cost': self.PRODUCT_COST,
        }
        for name, field in names.items():
            result[name] = list(map(itemgetter(field), query_result))
        return pd.DataFrame(result)

    def select_revenue_by_day_df(self):
        query_result = self.execute_query("""
            select date(date_time) as dt, sum(revenue) as SUM_REVENUE
            from orders
            group by dt
            order by dt asc
            """).fetchall()
        result = {}
        names = {
            'dt': 0,
            'sum_revenue': 1,
        }
        for name, field in names.items():
            result[name] = list(map(itemgetter(field), query_result))
        return pd.DataFrame(result)

    def get_start_date(self):
        return datetime.datetime(2018, 6, 1)

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
        logging.info('Cities inserted')

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
        logging.info('customer types inserted')

    def get_product_types(self):
        return {
            'coffee': ['samples', 'Tazza Low Lactose Hot Choco', 'Vendor Espresso', 'Paulig Special Medium beans',
                       'Tazza hot choco', 'Paulig Special Medium'],
            'coffeemachine': ['Thermoplan BLACK & WHITE', 'Nuova Simonelli Appia Life',
                              'Nuova Simonelli Aurelia Wave Digit', 'Jura E8 Platin Touch', 'Jura WE8'],
            'disposable': ['Vendor Whitener', 'paulig TA cup', 'Arla lact free portion milk',
                           'Paulig portionsugar brown'],
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
        logging.info('customers inserted')

    def insert_cafe(self, count: int = 100):
        customers_df = self.select_customers_df()
        city_df = self.select_city_df()

        for i in range(count):
            name = names.get_full_name()
            np.random.shuffle(list(name))
            name = ''.join(name)
            city_index = random.choice(range(len(city_df)), p=self.make_probabilities(len(city_df), [0.5, 0.2]))
            city_id = city_df.iloc[city_index]['idx']
            city_name = city_df.iloc[city_index]['name']
            address = ' {} street {}'.format(city_name, str(i))
            customer_index = random.choice(range(len(customers_df)))
            customer_id = customers_df.iloc[customer_index]['idx']
            self.execute_query(f"""INSERT INTO cafe VALUES
                              ('{name}', '{address}', '{customer_id}', '{city_id}')
                               """)
        logging.info('cafes inserted')

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

        costs = np.exp(random.poisson(4, count))
        for i in range(count):
            product_type_index = random.choice(range(len(product_types)), p=type_probabilities)
            product_type = product_types[product_type_index]
            product_type_names = product_names[product_types_names[product_type_index]]
            product_probabilities = self.make_probabilities(len(product_type_names))
            product_name = random.choice(product_type_names, p=product_probabilities)
            cost = np.round(min(costs[i], 1000), 2)
            self.execute_query(f"""INSERT INTO product VALUES
                              ('{product_name}', '{cost}', '{product_type}')
                               """)
        # for product in self.select_products():
        #     logging.info(product)
        logging.info('products inserted')

    def get_min_temp(self):
        return -10

    def generate_temperature(self, date: datetime.datetime):
        min_temp = self.get_min_temp()
        max_temp = 30

        day_of_year = date.timetuple().tm_yday
        shifts_count = 20
        local_shift = np.cos(day_of_year * np.pi * shifts_count / 365) * 2

        hottest_day = 365 / 2
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

        days_count = (end_date - cur_date).days

        prev_precip_prob = 0.5
        alpha = 0.8
        for i in tqdm.tqdm(range(days_count)):
            date_time = cur_date
            for i in range(count_per_day):
                date_time = date_time + datetime.timedelta(hours=random.randint(3, int(24 / count_per_day)))
                temperature = self.generate_temperature(date_time)
                wind_speed = abs(random.normal(10, 2))
                pressure = abs(random.normal(760, 2))
                humidity_logit = max(0, random.normal(2, 1))
                humidity = 1 / (1 + np.exp(-humidity_logit))
                precip_prob = 1 / (1 + np.exp(-(humidity_logit - 1 - temperature/20)))
                precip_prob = alpha * prev_precip_prob + (1 - alpha) * precip_prob
                prev_precip_prob = precip_prob
                self.execute_query(f"""INSERT INTO weather VALUES
                                  ('{date_time}', '{temperature}','{wind_speed}', '{pressure}', '{precip_prob}', '{humidity}')
                                   """)
            cur_date += datetime.timedelta(days=1)
        logging.info(cur_date)
        # for weather in self.select_weather()[-300:]:
        #     logging.info(weather)
        logging.info('weather inserted')

    def insert_orders(self, count_per_day=30):
        products_df = self.select_products_df()
        customers_df = self.select_customers_df()
        city_df = self.select_city_df()
        cafe_df = self.select_cafe_df()
        weather_df = self.select_weather_df()
        min_temp = self.get_min_temp()

        cur_date = self.get_start_date()
        end_date = self.get_end_date()
        if count_per_day > 8:
            count_per_day = 8

        days_count = (end_date - cur_date).days

        weather_idx = 0
        make_from_date = lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        weather_dates_str = weather_df['dt']
        weather_dates = list(map(make_from_date, weather_dates_str))
        weather_dates_cnt = len(weather_dates)

        customer_revenue_percents = {}

        for i in tqdm.tqdm(range(days_count)):
            date_time = cur_date + datetime.timedelta(hours=8)
            for i in range(count_per_day):
                date_time = date_time + datetime.timedelta(minutes=random.randint(10, int(16 * 60 / count_per_day)))
                product_index = random.choice(range(len(products_df)))
                product_type_id = products_df.iloc[product_index]['type']
                product_id = products_df.iloc[product_index]['idx']
                product_cost = float(products_df.iloc[product_index]['cost'])

                customer_index = random.choice(range(len(customers_df)))
                customer_id = customers_df.iloc[customer_index]['idx']

                if customer_id not in customer_revenue_percents:
                    revenue_percent = abs(random.normal(0.05, 0.01)) + 0.05
                    customer_revenue_percents[customer_id] = revenue_percent
                else:
                    revenue_percent = float(customer_revenue_percents[customer_id])

                revenue_per_item = np.round(product_cost * revenue_percent, 2)
                price = np.round(product_cost + revenue_per_item, 2)
                while weather_idx < weather_dates_cnt - 1 and date_time > weather_dates[weather_idx]:
                    weather_idx += 1
                weather_temp = abs(weather_df.iloc[weather_idx]['temp'] - min_temp)
                weather_precip_prob = weather_df.iloc[weather_idx]['precip_prob']
                weather_id = weather_df.iloc[weather_idx]['idx']

                weekend_additive = 0
                # if cur_date.weekday() >= 5:
                #     weekend_additive = 10
                good_weather_additive_qty = 15
                high_temp_additive_qty = 30
                quantity = weekend_additive \
                           + (1 - weather_precip_prob) * good_weather_additive_qty + high_temp_additive_qty * weather_temp / 10   # + int(1 + np.random.poisson(10, 1)[0])  \

                # if revenue_per_item > 20:
                #     quantity = min(quantity, 20)
                # if revenue_per_item > 100:
                #     quantity = min(quantity, 5)
                if revenue_per_item > 100:
                    quantity = min(1, quantity)
                revenue = np.round(revenue_per_item * quantity, 2)

                city_index = random.choice(range(len(city_df)))
                city_id = city_df.iloc[city_index]['idx']

                cafe_index = random.choice(range(len(cafe_df)))
                cafe_id = cafe_df.iloc[cafe_index]['idx']
                self.execute_query(f"""INSERT INTO orders VALUES
                                  ('{date_time}', '{quantity}', '{revenue}', '{price}', '{product_id}', '{product_type_id}', '{customer_id}', '{city_id}', '{cafe_id}', '{weather_id}')
                                   """)
            cur_date += datetime.timedelta(days=1)
        # for product in self.select_products():
        #     logging.info(product)
        logging.info('orders inserted')

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
                          cost real,
                          type integer,
                           FOREIGN KEY(type) REFERENCES product_type(ROWID))
                           """)
        self.execute_query("""CREATE TABLE weather
                          (date_time datetime,
                          temp REAL,
                          wind_speed REAL,
                          pressure REAL,
                          precip_prob REAL,
                          humidity REAL)
                           """)
        self.create_orders_table()
        logging.info('Tables recreated')

    def create_orders_table(self):
        # price per one piece
        # revenue = quantity * (price - cost)
        # denormalized scheme to avoid extra joins
        self.execute_query("""CREATE TABLE orders
                          (date_time datetime,
                          quantity integer,
                          revenue REAL,
                          price REAL, 
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
