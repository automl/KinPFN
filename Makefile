format:
	black .
	
test:
	black --check .
	pytest

lint:
	pylint --rcfile=./.pylintrc $$(find . -name '*.py' -not -path '*/\.*')