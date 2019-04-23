# tweetycs-flask

## Starting
After installing all the dependencies, create a JSON file in `config` folder and include the details of Twitter API:    
```bash
echo "{
    \"consumer_key\": \"SOMETHING\",
    \"consumer_secret\": \"SOMETHING\",
    \"access_token\": \"SOMETHING\",
    \"access_token_secret\": \"SOMETHING\"
}" >> config/config.json
```
Now Simply run the `app.py` file:
```bash
python app.py
```