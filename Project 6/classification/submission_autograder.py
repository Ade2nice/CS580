#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

