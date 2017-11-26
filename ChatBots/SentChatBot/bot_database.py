import sqlite3
import json
from datetime import datetime

timeframe = '2015-05'

# for building inserting entire transactions of data at once
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))

c = connection.cursor()

def create_table():
    # query
    c.execute('CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY), comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT')

def format_data(data):
    data = data.replace('\n', ' newlinechar').replace('"',"'")
    return data

def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1"
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        return False

if __name__=="__main__":
    create_table()
    # how many rows while iterating through files
    row_counter = 0
    # how many parent and child pairs there are
    paired_rows = 0
    # open one of the files
    with open('J:/chatdata/reddit_data/{}/RC_{}'.format(timeframe.split('-')[0],timeframe), buffer=1000) as f:
        for row in f:
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)
