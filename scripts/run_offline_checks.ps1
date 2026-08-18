$ErrorActionPreference = "Stop"

python -m compileall -q `
  .\app.py `
  .\logging_utils.py `
  .\src `
  .\practice `
  .\tests

python -m pytest .\tests -q
