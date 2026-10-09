.PHONY: install train test lint api demo docker

install:
	pip install -e ".[dev,app]"
train:
	python -m smsclf.train
test:
	pytest
lint:
	ruff check .
api:
	uvicorn smsclf.api:app --reload
demo:
	streamlit run app/streamlit_app.py
docker:
	docker compose up --build
