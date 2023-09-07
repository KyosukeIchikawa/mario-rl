VENV = .venv
PYTHON = python
REQUIREMENTS = requirements.txt

.PHONY: venv setup

venv:
	$(PYTHON) -m venv $(VENV)

setup: venv
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

clean:
	rm -rf $(VENV)
