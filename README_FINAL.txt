FINAL LOGIC
===========
YOLO output: Baby / Empty Seat / No Detection
Behaviour output: Inactive / Low Movement / Normal Movement / Restless Movement / Sudden Movement
AST output: Baby Cry / Noise / Normal
Fusion output: one of the 11 project fusion conditions
Risk output: mapped from the active fusion condition
SHAP graph: grouped YOLO, Behaviour, and AST contribution for the selected risk level
CSV: live_multimodal_risk_output.csv

RUN
===
& "C:\Users\Admin\Desktop\Multimodal_Fusion\fusion_env\Scripts\python.exe" -m py_compile .\app.py
& "C:\Users\Admin\Desktop\Multimodal_Fusion\fusion_env\Scripts\python.exe" .\app.py
