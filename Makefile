VENV = .venv
PYTHON = python
REQUIREMENTS = requirements.txt

.PHONY: venv setup test

venv:
	$(PYTHON) -m venv $(VENV)

setup: venv
	$(VENV)/bin/pip install -r $(REQUIREMENTS)
	$(VENV)/bin/pip install -e .

test:
	$(VENV)/bin/pytest tests/

clean:
	rm -rf $(VENV)
