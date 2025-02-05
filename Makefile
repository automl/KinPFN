format:
	black .
	
test:
	pytest

lint:
	pylint --rcfile=./.pylintrc $$(find . -name '*.py' -not -path '*/\.*')