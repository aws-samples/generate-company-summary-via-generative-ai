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
for company_URL, company_name in companies:
    pm.execute_notebook(
        "generate-company-summary.ipynb",
        Path.cwd() / "dossiers" / "notebooks" / f"{company_name}.ipynb",
        parameters=dict(company_URL=company_URL,
                        enable_test_harness=False))
