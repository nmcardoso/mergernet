install:
	python3 -m venv venv/
	chmod +x venv/bin/activate
	. venv/bin/activate; pip3 install -Ur requirements.txt

update:
	. venv/bin/activate; pip3 freeze > requirements.txt

