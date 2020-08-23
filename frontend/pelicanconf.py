#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

from datetime import datetime

from decouple import config

AUTHOR = "Srinivasan.G"
SITENAME = "RS Group"
SITETITLE = "RS Group"
SITEURL = config("SITE_URL")

PATH = "content"

TIMEZONE = "Asia/Kolkata"

DEFAULT_LANG = "en"

COPYRIGHT_YEAR = datetime.today().year
# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (("The School of AI", "https://theschoolof.ai/"),)

# Social widget
SOCIAL = (("Github", "https://github.com/seenu-g/eva4-2/"),)

DEFAULT_PAGINATION = 5

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

THEME = "phantom"
PLUGIN_PATHS = ["plugins"]
PLUGINS = ["pelican_javascript"]
