# [WACV 2025] Deep Geometric Moments Promote Shape Consistency in Text-to-3D Generation

### Instructions:
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Build the extension for Gaussian Splatting:
```
cd gs
./build.sh
```
3. Start training!
```python
python main.py --config-name=base prompt.prompt="<prompt>"
```
You can specify a different prompt for Point-E:
```python
python main.py --config-name=base prompt.prompt="<prompt>" init.prompt="<point-e prompt>"
```
4. Stage 2 Training
```python
python main.py --config-name=base prompt.prompt="<prompt>" init.prompt="<point-e prompt>" auxiliary.enabled=false renderer.densify.enabled=true renderer.prune.enabled=true
```
