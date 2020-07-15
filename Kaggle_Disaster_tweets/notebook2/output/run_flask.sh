directory_of_script="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $directory_of_script

export FLASK_APP=server.py
flask run
