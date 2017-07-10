import csv
import re
from datatype import *

def readUsers(path):
    users = {}
    with open(path, 'r') as f:
        csvf = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvf:
            # user_id,user_name,user_image,gender,class,message,post_num,follower_num,followee_num,is_spammer
            users[row[0]] = {
                'user_name' : row[1],
                'user_image' : row[2],
                'gender' : row[3],
                'class' : row[4],
                'message' : ''.join(row[5:-4]),
                'post_num' : row[-4],
                'follower_num' : row[-3],
                'followee_num' : row[-2],
                'is_spammer' : row[-1],
            }
    users.pop('user_id')
    print(len(users))
    return users
    
def readTweets(path, noUrl=True, noTopic=True, noAtUser=True):
    tweets = []
    re_url = re.compile('https?://[^/]+?/[\x00-\xff]*')
    re_topic = re.compile('#[^#]+#')
    re_at = re.compile('@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}')
    with open('data/user_posts.csv', 'r') as f:
        csvf = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvf:
            
            flag = False
        
            # if void
            if not row[0]:
                continue
        
            # check stop-words
    #         for word in stopWords:
    #             if word in row[2]:
    #                 flag = True
    #                 break
                
            # check num of words
            content = row[2]
            content = re_url.sub('', content) if noUrl else content
            content = re_topic.sub('', content) if noTopic else content
            content = re_at.sub('', content) if noAtUser else content
            if len(content) < 10:
                flag = True
        
            # check 0 comment
            try :
                if int(row[-3]) == 0:
                    flag = True
            except ValueError:
                print('!!' , row)
                flag = True
        
            # 
            if flag:
                continue
            
            tweets.append(Tweet(
                post_time = row[1],
                content = content,
                poster_id = row[-6],
                poster_url = row[-5],
                repost_num = row[-4],
                comment_num = int(row[-3]),
            ))
    print(len(tweets))
    # tweets[0:5]
    # users.get(tweets[99].poster_id)
    return tweets
    
    