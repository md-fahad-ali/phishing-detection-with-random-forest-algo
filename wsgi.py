import sys
import os

# Add your project directory to the sys.path
project_home = '/home/your_pythonanywhere_username/mysite'
if project_home not in sys.path:
    sys.path.append(project_home)

# Import your Flask app
from phishing_api import app as application

# This is the PythonAnywhere WSGI configuration
if __name__ == '__main__':
    application.run()
