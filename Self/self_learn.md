1. # If new created Ipykernel did not show
Register ipykernel:python -m ipykernel install --user --name=hw3 --display-name "Python (HW3)"
 Ctrl+Shift+P : reload window, kernel refresh
 run: jupyter kernelspec list / jupyter kernelspec list --json
 cd kernel path
 check with nano kernel.json
 force to apply :python -c "import sys; print(sys.executable)"
2. 
# why asyncio.get_running_loop() instead of asyncio.run(_crawl())
Because in Jupyter / IPython, there’s already a running event loop → so you get errors like
RuntimeError: This event loop is already running
asyncio.get_running_loop() is a workaround 
3.
# nest_asyncio allows nested event loops
Normally, Python forbids running a loop inside another loop
4.
# pip install google-genai