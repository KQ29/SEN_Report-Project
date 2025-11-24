# Student Report Generator (Streamlit)

## Run locally
pip install -r requirements.txt
streamlit run streamlit_app.py

## Qwen2.5-0.5B-Instruct (optional LLM helper)
1. Ensure the virtual environment is active: `source env/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the weights (already checked into `models/Qwen2.5-0.5B-Instruct`, or run `python qwen_integration.py "ping"` to fetch automatically).
4. Use `python qwen_integration.py "Your prompt here"` to generate a quick response locally.

## Data
Place `complete_dummy_data.json` in the same folder or use the sidebar path.
