
echo "Enter commit message: "
read -r message
echo "The commit message is: $message"
find ./ -name ".ipynb_checkpoints" -exec rm -R {} \;
find ./ -name "__pycache__" -exec rm -R {} \;
git add .
git commit -am "$message"
git push origin master
