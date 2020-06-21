directory_of_script="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $directory_of_script
jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
