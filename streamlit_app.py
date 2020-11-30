import streamlit as st
st.set_page_config(page_title="New York City", layout='wide', initial_sidebar_state='collapsed')

import os
import time
import datetime
from functools import reduce

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
px.set_mapbox_access_token("pk.eyJ1IjoieHh4OTgiLCJhIjoiY2tocmxlYXJpMHZmdzJycnNuM2J3eW1weCJ9.O3Ep-9NHHtcU4lTKoxXXOg")

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

hash_funcs={'_thread.RLock' : lambda _: None, 
                        '_thread.lock'    : lambda _: None, 
                        'pyspark.broadcast.BroadcastPickleRegistry' : lambda _ :None,
                        'py4j.java_gateway.JVMView' : lambda _ : None}
factors_columns = ['CONTRIBUTING FACTOR VEHICLE '+str(i) for i in range(1, 6)]
factors_columns_new = ['factor_'+str(i) for i in range(1, 6)]

vehicles_columns = ['VEHICLE TYPE CODE '+str(i) for i in range(1, 6)]
vehicles_columns_new = ['vehicle_'+str(i) for i in range(1, 6)]

def __numbers_columns(df, camel_case=True):
    return [column.title() if camel_case else column for column in df.columns if column.startswith('NUMBER OF')]

@st.cache(allow_output_mutation=True, hash_funcs=hash_funcs)                    
def __init():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    df = spark.read.format("csv").option("inferSchema", True).option("header", True).load("./data.csv")
    return df

def __value_for_columns(data, columns):
    results = {}
    for column in columns:
        for key, count in data.select(column).groupBy(column).count().collect():
            results[key] = results.get(key, 0) + count
    return [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

def __get_data_for_sliders(df):
    unique_data = {}
    unique_data['BOROUGH'] = __value_for_columns(df, ["BOROUGH"])

    unique_data['FACTORS'] = __value_for_columns(df, factors_columns_new)
    unique_data['VEHICLES'] = __value_for_columns(df, vehicles_columns_new)
    for column in __numbers_columns(df):
        unique_data[column] = df.agg({column: "max"}).collect()[0][0]
    return unique_data

@st.cache(allow_output_mutation=True, hash_funcs=hash_funcs)
def __get_filtered_data():
    df = __init()

    for old_name_columns, new_name_columns in zip([factors_columns, vehicles_columns], 
                                                  [factors_columns_new, vehicles_columns_new]):
        for old_name, new_name in zip(old_name_columns, new_name_columns):
            df = df.withColumnRenamed(old_name, new_name)

    numbers = __numbers_columns(df, False)
    interested_columns = ['LATITUDE', 'LONGITUDE', 'CRASH DATE', 'BOROUGH', 'CRASH TIME']+factors_columns_new+vehicles_columns_new+numbers
    nullable_columns = factors_columns_new+vehicles_columns_new+['BOROUGH']

    df = df.drop(*[column for column in df.columns if column not in interested_columns])
    for column in interested_columns:
        if column not in nullable_columns:
            df = df.where(col(column).isNotNull())
        else:
            df = df.withColumn(column, when(col(column).isNull(), "Unspecified").otherwise(col(column)))
        
        if column in numbers:
            df = df.withColumn(column, df[column].cast("int"))
            df = df.withColumnRenamed(column, column.title())

        df = df.withColumn(column, initcap(col(column)))

    df = df.where((col('LATITUDE') != 0) &
                                (col('LONGITUDE') != 0) &
                                (col('LONGITUDE') >= -100))
    
    df = df.withColumn('year', year(to_date('CRASH DATE', 'MM/dd/yyy')))
    df = df.withColumn('hour', hour('CRASH TIME'))
    df = df.drop('CRASH DATE').drop('CRASH TIME')
    df = df.orderBy("year", ascending=True)
    return df, __get_data_for_sliders(df)

def get_cached_data():
    start = time.time()
    df, dict_data_for_sliders = __get_filtered_data()
    stop = time.time()
    st.write('Time to load initial data: ' + str(stop-start))
    return df, dict_data_for_sliders

def fill_config(unique_data):
    st.sidebar.title("Config")
    unique_data['SHOW_COUNT'] = st.sidebar.checkbox('Show count after each block?', False)
    unique_data['EXPANDED'] = st.sidebar.checkbox("Expanded by default", False)
    return unique_data

def __filter_by_columns(df, unique_values, columns, header, selector_text, sidebar = False):
    with st.beta_expander(header, unique_data['EXPANDED']):
        selected = unique_values if st.checkbox('Force all values', key = header) else unique_values[:10]
        result = st.multiselect(selector_text, unique_values, selected)
        df = df.filter(reduce(lambda x, y: x | y, (col(name).isin(result) for name in columns)))
        if unique_data['SHOW_COUNT']:
            st.markdown(f'Count of values after this block: {df.count()}')
    return df

def select_interested_regions(df, unique_data):
    return __filter_by_columns(df, unique_data['BOROUGH'], ['BOROUGH'], "Regions of New York", "Which regions to show?")

def filter_by_time(df, unique_data):
    with st.beta_expander("Range of time", unique_data['EXPANDED']):
        include = st.checkbox('Range to include', True)
        hours = st.slider("Select interested region of hours:", value=(datetime.time(0,0), datetime.time(23,59)),step=datetime.timedelta(hours=1))
        interested_hours = list(range(hours[0].hour, hours[1].hour+1))

        if not include:
            interested_hours = [v for v in range(0,24) if v not in interested_hours]

        st.write(f"You're selected hours: {interested_hours}")
        df = df.filter(col('hour').isin(interested_hours))
        if unique_data['SHOW_COUNT']:
            st.markdown(f'Count of values after this block: {df.count()}')
    return df

def filter_by_factor(df, unique_data):
    return __filter_by_columns(df, unique_data['FACTORS'], factors_columns_new, "Contributing factors vehicle", "Which factors to show?")

def filter_by_vehicle(df, unique_data):
    return __filter_by_columns(df, unique_data['VEHICLES'], vehicles_columns_new, "Vehicle type", "Which vehicle types to select?")

def filter_by_number_columns(df, unique_data):
    def filter_by_number(df, column, max_value):
        with st.beta_expander(column, unique_data['EXPANDED']):
            min_v, max_v = st.slider("Which region is interested?", min_value=0, max_value=max_value, value=(0,  max_value), key=column)
            df = df.filter(df[column] >=min_v).filter(df[column] <= max_v)
            if unique_data['SHOW_COUNT']:
                st.markdown(f'Count of values after this block: {df.count()}')
        return df

    numbers = __numbers_columns(df)
    columns = st.beta_columns(len(numbers))
    for i, column in enumerate(numbers):
        with columns[i]:
            df = filter_by_number(df, column, int(unique_data[column]))
    return df

def cut_data_by_slider(df, unique_data):
    with st.beta_expander("Size of data", unique_data['EXPANDED']):
        percent = st.slider("Percent of data to use", 1, 100, 75)
        percent = percent/100
        df = df.sample(False, percent)
        if unique_data['SHOW_COUNT']:
            st.markdown(f'Count of values after this block: {df.count()}')
    return df
    
def plot_points(dataframe, time_slider=False):
    temp_dataframe = dataframe.select(['LATITUDE', 'LONGITUDE', 'year']).toPandas()
    fig = px.scatter_mapbox(temp_dataframe, lat='LATITUDE', lon='LONGITUDE', hover_name='year', zoom=10, animation_frame='year' if time_slider else None)
    fig.update_layout(mapbox_style="open-street-map", height=800)

    st.plotly_chart(fig, use_container_width=True)

##################### MAIN CODE ##########################
st.title("New York City Map")
df, unique_data = get_cached_data()

unique_data = fill_config(unique_data)

#st.write(df)
#st.write(df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).toPandas())

st.header('Configurate values')

df = select_interested_regions(df, unique_data)
df = filter_by_time(df, unique_data)
df = filter_by_factor(df, unique_data)
df = filter_by_vehicle(df, unique_data)
df = filter_by_number_columns(df, unique_data)

df = cut_data_by_slider(df, unique_data)

timeline = st.checkbox('Enable timeline')
st.write('Total count of points: ' + str(df.count()))
if st.checkbox('Autoplot') or st.button("Plot it!"):
    with st.spinner('Plotting...'):
        plot_points(df, timeline)