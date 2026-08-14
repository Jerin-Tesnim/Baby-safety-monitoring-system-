LIVE SHAP FINAL VIVA DASHBOARD

This package contains:
- app.py
- templates/index.html
- best_tree_model_for_shap.pkl
- preflight_check.py

The app uses the existing model paths already configured in app.py:
YOLO: C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat\runs\detect\baby_person_empty_seat\weights\best.pt
LSTM files: C:\Users\Admin\Desktop\MediaPipe_33_Landmarks\
AST model: C:\Users\Admin\Desktop\AST_Audio_Project\ast_audio_model

RUN
1. Open PowerShell in this extracted folder.
2. Test syntax:
   & "C:\Users\Admin\Desktop\Multimodal_Fusion\fusion_env\Scripts\python.exe" -m py_compile .\app.py
3. Run:
   & "C:\Users\Admin\Desktop\Multimodal_Fusion\fusion_env\Scripts\python.exe" .\app.py
4. Open http://127.0.0.1:5000
5. For another network, open another terminal and run: ngrok http 5000

OUTPUT
- Live camera with YOLO box and MediaPipe landmarks
- YOLO, Behaviour, AST Audio, and Risk cards
- Three grouped native-XGBoost SHAP bars: YOLO, Behaviour, AST
- One of the 11 fusion conditions and a short explanation
- live_multimodal_risk_output.csv generated automatically
