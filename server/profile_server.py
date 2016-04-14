#! /usr/bin/env python3

from werkzeug.contrib.profiler import ProfilerMiddleware
import server
from server import app

app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
app.run(debug = True)
