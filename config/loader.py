import json
import os

"""
    :param app: Flask Object
"""


def load_config(app):
    try:
        with open(os.path.dirname(__file__) + '/config.json') as json_file:
            obj = json.loads(json_file.read())
            app.config.update(obj)
    except Exception as e:
        print(e)
    finally:
        return app
