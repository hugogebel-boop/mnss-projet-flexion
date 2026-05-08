@echo off
setlocal
cd /d "%~dp0"
pdflatex -shell-escape -interaction=nonstopmode rapport.tex
pdflatex -shell-escape -interaction=nonstopmode rapport.tex
echo.
echo Done. PDF: %CD%\rapport.pdf
endlocal
