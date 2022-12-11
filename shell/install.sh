sudo apt install -y python3-pip
sudo apt install -y python3-virtualenv
python3 -m venv venv
source venv/bin/activate
pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install -r requirements.txt
