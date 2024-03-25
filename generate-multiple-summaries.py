"""
Run the Jupyter notebook as a batch job.
"""
from pathlib import Path

import papermill as pm

companies = [
    ("https://www.agios.com", "agios"),
    ("https://www.morphosys.com/en", "morphosys"),
    ("https://incyte.com", "incyte"),
    ("https://www.protagonist-inc.com", "protagonist"),
    ("https://kerostx.com", "keros"),
    ("https://vaderis.com", "vaderis"),
    ("https://hemavant.com", "hemavant"),
    ("https://www.intelliatx.com", "intellia"),
    ("https://beamtx.com", "beam"),
    ("https://www.capstantx.com", "capstan"),
    ("https://renagadetx.com", "renagade"),
    ("https://arbor.bio", "arbor")
]
notebooks_dir = Path.cwd() / "summaries" / "notebooks" 
notebooks_dir.mkdir(parents=False, exist_ok=True)
for company_URL, company_name in companies:
    pm.execute_notebook(
        "generate-company-summary.ipynb",
        notebooks_dir / f"{company_name}.ipynb",
        parameters=dict(company_URL=company_URL,
                        enable_test_harness=False))
