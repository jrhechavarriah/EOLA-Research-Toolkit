@echo off
setlocal

cd /d "%~dp0"

for %%D in (
    config
    data
    data\raw
    data\interim
    data\processed
    data\metadata
    docs
    figures
    models
    notebooks
    outputs
    reports
    release
    src
    tests
) do (
    if not exist "%%D" mkdir "%%D"
)

echo EOLA project structure ready at:
cd

endlocal