.PHONY: install run clean app

install:
	pip install -r requirements.txt

run:
	python scripts/run_pipeline.py --config config/config.yaml

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed/* reports/figures/* reports/tables/* reports/logs/*
