from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load não AI và Thư viện kiến thức
try:
    payload = joblib.load('blood_ai_ultimate.pkl')
    model = payload['model']
    features_list = payload['features']
    clinical_db = payload['knowledge']
except Exception as e:
    print(f"LỖI: Không tìm thấy file blood_ai_ultimate.pkl. Chi tiết: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df_in = pd.DataFrame([data])
    
    # 1. Tự động bù đắp các chỉ số thiếu
    defaults = {'HCT': 40, 'MCH': 30, 'MCHC': 33, 'Monocytes': 5, 'RDW': 13}
    for col in ['HCT', 'MCH', 'MCHC', 'Monocytes', 'RDW']:
        if col not in df_in.columns:
            df_in[col] = defaults[col]

    # 2. Feature Engineering
    df_in['Mentzer_Index'] = df_in['MCV'] / (df_in['RBC'] + 0.0001)
    df_in['NLR'] = df_in['Neutrophils'] / (df_in['Lymphocytes'] + 0.0001)
    df_in['PLR'] = df_in['PLT'] / (df_in['Lymphocytes'] + 0.0001)

    # 3. AI Dự đoán
    probs = model.predict_proba(df_in[features_list])
    target_names = ['Infection', 'Anemia', 'Bleeding', 'Deficiency']
    
    # 4. Lọc và Xếp hạng Top 2
    all_predictions = []
    for i, name in enumerate(target_names):
        prob_val = probs[i][0][1]
        all_predictions.append({"name": name, "prob_val": prob_val})
        
    all_predictions.sort(key=lambda x: x['prob_val'], reverse=True)
    top_risks = [p for p in all_predictions if p['prob_val'] > 0.25][:2]
    
    # 5. Soạn báo cáo biện luận
    final_report = []
    for idx, p in enumerate(top_risks):
        name = p['name']
        prob_val = p['prob_val']
        info = clinical_db[name]
        
        impact_details = [{"name": m, "val": round(float(df_in[m].iloc[0]), 2)} for m in info['key_indicators']]
        diagnosis_type = "Chẩn đoán chính" if idx == 0 else "Nguy cơ đồng mắc"

        final_report.append({
            "category": f"[{diagnosis_type}] {info.get('title', name)}",
            "probability": f"{prob_val*100:.1f}%",
            "risk_level": "Cao" if prob_val > 0.7 else "Trung bình",
            "mechanism": info['mechanism'],
            "potential_causes": info['potential_causes'],
            "impact_factors": impact_details,
            "scientific_evidence": info['scientific_evidence']
        })

    return jsonify(final_report)

if __name__ == '__main__':
    app.run(debug=True, port=5000)