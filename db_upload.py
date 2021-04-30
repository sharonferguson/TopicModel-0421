
import mysql.connector
import json
import os
import time
from datetime import datetime 
from pathlib import Path

##This file reads all of the json files and uploads them to a SQL workbench (MySQL database). To create the database for the first time use the first db connection and uncomment out the create table lines.
#if you are running it after the database is created use the second db connection and comment out the create table lines

def createDB():
    '''use this function if you have not created the database yet. This will create the database and the tables'''
    mydb = mysql.connector.connect(user='root',
                                password = 'SharonF1.',
                                host='localhost',
                                port = '3306',
                                database = 'sys',
                                #auth_plugin='mysql_native_password',
                                #raw = False
    )

    c = mydb.cursor(buffered=True)

    c.execute("CREATE DATABASE slackdatabase")

    c.execute("USE slackdatabase;")

    # #create tables 

    #table 1 - team
    c.execute("CREATE TABLE team (id INT AUTO_INCREMENT PRIMARY KEY, year int, name VARCHAR(255), source VARCHAR(255))")

    #table 2 - user
    c.execute("CREATE TABLE user (slack_uid VARCHAR(255) PRIMARY KEY, id_team int, role VARCHAR(255), admin BOOLEAN, bot BOOLEAN, gender VARCHAR(255), ethnicity VARCHAR(255), first_gen BOOLEAN, FOREIGN KEY (id_team) REFERENCES team(id))")

    #table 3 - channel
    c.execute("CREATE TABLE channel (slack_cid VARCHAR(255) PRIMARY KEY, name VARCHAR(255), description VARCHAR(255), id_team int, num_pins int, created datetime, createdby VARCHAR(255), archived BOOLEAN, FOREIGN KEY (id_team) REFERENCES team(id), FOREIGN KEY (createdby) REFERENCES user(slack_uid))")

    #table 4 - userchannellink
    c.execute("CREATE TABLE userchannellink (id INT AUTO_INCREMENT PRIMARY KEY, id_user VARCHAR(255), id_channel VARCHAR(255), id_team int, FOREIGN KEY (id_user) REFERENCES user(slack_uid), FOREIGN KEY(id_channel) REFERENCES channel(slack_cid), FOREIGN KEY (id_team) REFERENCES team(id))")

    #table 5 - message
    c.execute("CREATE TABLE message (id INT AUTO_INCREMENT PRIMARY KEY, id_user VARCHAR(255), id_channel VARCHAR(255), comment TEXT, timestamp datetime, reacts BOOLEAN, replies BOOLEAN, reply_count int, reply_usercount int, attachment BOOLEAN, edited BOOLEAN, subtype VARCHAR(255), mentions VARCHAR(255), FOREIGN KEY (id_user) REFERENCES user(slack_uid), FOREIGN KEY(id_channel) REFERENCES channel(slack_cid))")

    #table 6 - react
    c.execute("CREATE TABLE react (id INT AUTO_INCREMENT PRIMARY KEY, id_user VARCHAR(255), id_message int, name VARCHAR(255), FOREIGN KEY (id_user) REFERENCES user(slack_uid), FOREIGN KEY(id_message) REFERENCES message(id))")

    #table 7 - replies
    c.execute("CREATE TABLE replies (id INT AUTO_INCREMENT PRIMARY KEY, id_user VARCHAR(255), parent_user_id VARCHAR(255), thread_timestamp datetime,  id_replymessage int, reply_message TEXT, reply_timestamp datetime, FOREIGN KEY(id_user) REFERENCES user(slack_uid))")

def connectDB():
    mydb = mysql.connector.connect(user='root',
                                password = 'SharonF1.',
                                host='localhost',
                                port = '3306',
                                database = 'slackdatabase',
                                auth_plugin='mysql_native_password',
                                raw = False
    )

    c = mydb.cursor(buffered=True)

    

    return mydb, c


def loadTeam(mydb, c, year, team, name, source):
    
    start_time = time.time()

    query = "INSERT INTO team (year,name,source) VALUES (%s,%s,%s)" 
    values = (year, team, source)
    c.execute(query, values)
    tid = c.lastrowid

    mydb.commit()
    team_id = c.lastrowid

    users = dict()
    channels = dict()
    channel_membership = dict()

    with open("/Users/sharon/Documents/SlackData/" + str(year) + "/" + name + "users.json") as file:
        data = json.load(file)

        #get all users
        for user in data:

            if user['deleted']  == False:
                uid2 = c.lastrowid
                users[user["id"]] = uid2
                try:
                    c.execute('INSERT INTO user (`id_team`, `slack_uid`, `admin`, `bot`) VALUES (%s,%s,%s, %s)', (team_id, user['id'], int(user['is_admin']), (user['is_bot'])))
                except mysql.connector.errors.IntegrityError as e:
                    print("Error: {}".format(e))
    mydb.commit()
    #get all channels
    with open("/Users/sharon/Documents/SlackData/" + str(year) + "/" + name + "channels.json") as file:
        data = json.load(file)

        for channel in data:
            cname = channel['name']
            cid =channel['id']
            ts = datetime.fromtimestamp(int(float(channel['created'])))
            archived = int(channel['is_archived'])
            desc = channel['purpose']['value']
            createdby = channel['creator']
            if "pins" in channel:
                num_pins = len(channel["pins"])
            else:
                num_pins = 0 
            c.execute('INSERT INTO channel (`id_team`, `name`, `slack_cid`, `archived`, `created`, `description`, `createdby`, `num_pins` ) VALUES (%s,%s,%s,%s,%s,%s, %s, %s)', (team_id, cname, cid, archived, ts, desc, createdby, num_pins))
            channels[cname] = cid
            channel_membership[cid] = channel['members']
    mydb.commit()
    #get all messages in each channel-day file
    for folder in os.listdir("/Users/sharon/Documents/SlackData/" + str(year) + "/" + name):
        if os.path.isdir("/Users/sharon/Documents/SlackData/" + str(year) + "/" + name + folder):
            fpath = "/Users/sharon/Documents/SlackData/" + str(year) + "/"+ name + folder + "/"
            for day in os.listdir(fpath):
                path = fpath + day
                if os.stat(path).st_size == 0: #checks if empty
                    continue 
                elif path[-1] != 'n': #checks for json file
                    continue
                with open(path) as file:
                    print(file)
                    data = json.load(file)

                    cid = channels[folder]

                    for msg in data:
                        if msg.get('subtype') and msg['subtype'] == "channel_join":
                            if msg['user'] not in users:
                                uid2 = c.lastrowid
                                users[msg["user"]] = uid2
                                try:
                                    c.execute('INSERT INTO user (`id_team`, `slack_uid`) VALUES (%s,%s)', (team_id, msg['user']))
                                except mysql.connector.errors.IntegrityError as e:
                                    print("Error: {}".format(e))
                            if msg['user'] not in channel_membership[cid]:
                                channel_membership[cid].append(msg['user'])
                        elif msg.get('subtype') and msg['subtype'] == "file_share":
                            subtype = msg['subtype']
                            file_id = msg['file']['id']
                            if msg['user'] not in users:
                                continue
                            uid = msg['user']
                            num_replies = int(msg['file']['comments_count'])
                            if num_replies >= 1:
                                replies = True
                            else:
                                replies = False
                            ts = datetime.fromtimestamp(int(float(msg['ts'])))
                            attachment = True
                            if msg['file']['comments_count'] >0 and msg['file'].get('initial_comment'):
                                comment = msg['file']['initial_comment']['comment']
                            else:
                                comment = ""
                            if msg.get('reactions'):
                                reactions = True 
                            else:
                                reactions = False
                            if msg.get('edited'):
                                edited = True
                            else:
                                edited = False
                            if "<!" in comment:
                                mentions = "channel"
                            elif "<@" in comment:
                                mentions = "individual"
                            else:
                                mentions = "None"
                            c.execute('INSERT INTO message (`id_user`, `id_channel`, `comment`, `timestamp`, `reacts`,`replies`, `reply_count`, `reply_usercount`, `attachment`, `edited`, `subtype`, `mentions`) VALUES (%s,%s,%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)', (uid, cid, comment, ts, reactions, replies, num_replies, 0, True, edited, subtype, mentions))
                            mid = c.lastrowid 
                            if msg.get('reactions'):
                                for react in msg['reactions']: 
                                    name = react['name']
                                    for name in react['users']:
                                        c.execute('INSERT INTO react (`id_user`, `id_message`, `name`) VALUES (%s, %s, %s)', (uid, mid, name))
                        elif msg.get('subtype') and msg['subtype'] == "file_comment":
                            subtype = msg['subtype']
                            fc = msg['comment']
                            uid = fc['user']
                            ts = datetime.fromtimestamp(int(float(fc['timestamp'])))
                            comment = fc['comment']
                            if fc.get('reactions'):
                                reactions = True
                            if fc.get('edited'):
                                edited = True
                            if "<!" in comment:
                                mentions = "channel"
                            elif "<@" in comment:
                                mentions = "individual"
                            else:
                                mentions = "None"
                            c.execute('INSERT INTO message (`id_user`, `id_channel`, `comment`, `timestamp`, `reacts`,`replies`, `reply_count`, `reply_usercount`, `attachment`, `edited`, `subtype`, `mentions`) VALUES (%s,%s,%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)', (uid, cid, comment, ts, reactions, False, 0, 0, False, edited, subtype, mentions))
                        elif msg.get('subtype') and msg['subtype'] == "reminder_add":
                            if msg['user'] not in users:
                                continue
                            subtype = msg['subtype']
                            comment = msg['text']
                            uid = msg['user']
                            ts = datetime.fromtimestamp(int(float(msg['ts'])))
                            if msg.get('reactions'):
                                reactions = True 
                            if "<!" in comment:
                                mentions = "channel"
                            elif "<@" in comment:
                                mentions = "individual"
                            else:
                                mentions = "None"
                            c.execute('INSERT INTO message (`id_user`, `id_channel`, `comment`, `timestamp`, `subtype`, `reacts`, `mentions`) VALUES (%s,%s,%s,%s, %s, %s, %s)', (uid, cid, comment, ts, subtype, reactions, mentions))
                            if msg.get('reactions'):
                                for react in msg['reactions']: 
                                    name = react['name']
                                    for name in react['users']:
                                        c.execute('INSERT INTO react (`id_user`, `id_message`, `name`) VALUES (%s, %s, %s)', (uid, mid, name))
                        else:
                            if not 'user' in msg:
                                continue
                            if msg['user'] == "USLACKBOT":
                                continue
                            elif msg['user'] not in users:
                                uid2 = c.lastrowid
                                users[msg["user"]] = uid2
                                try:
                                    c.execute('INSERT INTO user (`id_team`, `slack_uid`) VALUES (%s,%s)', (team_id, msg['user']))
                                except mysql.connector.errors.IntegrityError as e:
                                    print("Error: {}".format(e))
                            if msg['user'] not in channel_membership[cid]:
                                channel_membership[cid].append(msg['user'])
                            if msg.get('subtype'):
                                subtype = msg['subtype']
                            else:
                                subtype = "message"
                            uid = msg['user']
                            reactions = int(msg.get('reactions') != None)
                            replies = int(msg.get('replies') != None)
                            if replies != 0:
                                replies = True
                                reply_count = msg['reply_count']
                                if msg.get('reply_users_count'):
                                    reply_usercount = msg['reply_users_count']
                                else: 
                                    reply_usercount = 0
                            else:
                                replies = False
                                reply_count = 0
                                reply_usercount = 0
                            if msg.get('edited'):
                                edited = True
                            else: 
                                edited = False
                            if msg.get('attachments') or msg.get('file') or msg.get('files'):
                                attachment = True
                            else:
                                attachment = False
                            ts = datetime.fromtimestamp(int(float(msg['ts'])))
                            comment = msg['text']
                            if "<!" in comment:
                                mentions = "channel"
                            elif "<@" in comment:
                                mentions = "individual"
                            else:
                                mentions = "None"
                            c.execute('INSERT INTO message (`id_user`, `id_channel`, `comment`, `timestamp`, `reacts`,`replies`, `reply_count`, `reply_usercount`, `attachment`, `edited`, `subtype`, `mentions`) VALUES (%s,%s,%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)', (uid, cid, comment, ts, reactions, replies, reply_count, reply_usercount, attachment, edited, subtype, mentions))
                            mid = c.lastrowid
                            #now we need to test if it is a reply message, if it is we add it to the replies table 
                            if msg.get('parent_user_id'):
                                #leave original message id and text for a table join at the end, take out foreign key for id_message 
                                uid = msg['user']
                                thread_ts = datetime.fromtimestamp(int(float(msg['thread_ts'])))
                                parent_uid = msg['parent_user_id']
                                comment = msg['text']
                                reply_ts = datetime.fromtimestamp(int(float(msg['ts'])))
                                id_reply_message = mid
                                c.execute('INSERT INTO replies (`id_user`, `parent_user_id`, `thread_timestamp`, `id_replymessage`, `reply_message`, `reply_timestamp`) VALUES (%s, %s, %s, %s, %s, %s)', (uid, parent_uid, thread_ts, id_reply_message, comment, reply_ts))
                            elif msg.get('subtype') and msg['subtype'] == "thread_broadcast" and msg['root'].get('subtype') and msg['root']['subtype'] != 'bot_message':
                                uid = msg['user']
                                thread_ts = datetime.fromtimestamp(int(float(msg['thread_ts'])))
                                parent_uid = msg['root']['user']
                                comment = msg['text']
                                reply_ts = datetime.fromtimestamp(int(float(msg['ts'])))
                                id_reply_message = mid
                                c.execute('INSERT INTO replies (`id_user`, `parent_user_id`, `thread_timestamp`, `id_replymessage`, `reply_message`, `reply_timestamp`) VALUES (%s, %s, %s, %s, %s, %s)', (uid, parent_uid, thread_ts, id_reply_message, comment, reply_ts))
                            if reactions == 1:
                                for r in msg['reactions']:
                                    for u in r['users']:
                                        uid = u
                                        c.execute('INSERT INTO react (`id_user`, `id_message`, `name`) VALUES (%s,%s,%s)', (uid, mid, r['name']))

    mydb.commit()
    
     #all all channel member links to table
    for cid in channel_membership.keys():
        for uid in channel_membership[cid]:
            try:
                c.execute('INSERT INTO userchannellink (`id_user`, `id_channel`, `id_team`) VALUES (%s,%s, %s)', (uid, cid, tid))
            except mysql.connector.errors.IntegrityError as e:
                print("Error: {}".format(e))
    mydb.commit()

    elapsed_time = time.time() - start_time
    print(elapsed_time)

years = [2016, 2017, 2018, 2019, 2020]
teams = ["blue", "green", "orange", "pink", "purple", "red", "silver", "yellow"]

def populateDB(years, teams):
    #create DB
    createDB()
    #connect to the DB
    mydb, c = connectDB()

    #populate the DB
    for year in years:
        for team in teams:
            name = team + "/" + team + "/"
            if year != 2020:
                source = "MIT 2.009"
            if year == 2020:
                source = "MIT 2.s009"
            path2 = "/Users/sharon/Documents/SlackData/" + str(year) + "/" + name + "users.json"
            print(path2)
            if os.path.exists(path2):
                loadTeam(mydb, c, year, team, name, source)
    #create complete replies table
    #create the CompleteReplies merged table from a query  (this joins the message and replies table so that we can see the original message and all of its replies neatly in one table)
    c.execute('CREATE TABLE CompleteReplies SELECT r.id, m.id as original_id, m.comment, r.id_user, r.parent_user_id, r.thread_timestamp, r.id_replymessage, r.reply_message, r.reply_timestamp FROM message m inner join replies r ON m.id_user = r.parent_user_id AND m.timestamp = r.thread_timestamp WHERE m.replies = 1;')

    c.close()
    mydb.close()

populateDB(years, teams) 