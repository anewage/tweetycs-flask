import tweepy
import pymongo
import faust

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["tweetSense"]
mycol = mydb["tweets"]

consumer_key = "Twpe1G9ImNRKSrqHrBrjmybOx"
consumer_secret = "xgXksq11ncSZCMJfE6qcMPbzCliXHfTRvkMK3halPDpadfmAsL"
access_token = "563293530-2JwNmatiH4vlGLjLLG8529vH4y62OV7LPq3HugDr"
access_token_secret = "vfp3VMgyQQEzNnc1MqaAJ6MMokcv1HiHhlbZWuv1hjgju"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        x = mycol.insert_one(status._json)
        # print list of the _id values of the inserted documents:
        print(x)


stream_listener = MyStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=["trump", "clinton", "hillary clinton", "donald trump"], is_async=True)

# myquery = { "sentiment_score": -0.817562 }
# mydoc = mycol.find(myquery)