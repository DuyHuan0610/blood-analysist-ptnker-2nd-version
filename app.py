import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =================================================================
# 1. THƯ VIỆN KIẾN THỨC Y KHOA CHUYÊN SÂU (EXPLANATION ENGINE)
# =================================================================
CLINICAL_KNOWLEDGE = {
    "Infection": {
        "title": "Hội chứng Nhiễm trùng / Viêm cấp tính",
        "mechanism": "Hệ miễn dịch huy động bạch cầu (đặc biệt là bạch cầu trung tính) để thực bào và tấn công tác nhân ngoại lai hoặc đáp ứng với tổn thương mô.",
        "potential_causes": "Nhiễm khuẩn cấp tính hoặc Nhiễm virus (Cúm, Sốt xuất huyết giai đoạn đầu) hoặc Phản ứng viêm hệ thống (Sepsis) hoặc Stress sinh lý sau chấn thương nặng.",
        "scientific_evidence": {
            "source": "Mayo Clinic Laboratories (2023)",
            "quote": "Tăng bạch cầu (Leukocytosis > 11.0 K/uL) là chỉ báo lâm sàng chính của phản ứng viêm và nhiễm trùng.",
            "url": "https://www.mayocliniclabs.com/"
        },
        "key_indicators": ["WBC", "Neutrophils", "NLR"]
    },
    "Anemia": {
        "title": "Hội chứng Thiếu máu (Anaemia)",
        "mechanism": "Sự suy giảm nồng độ Hemoglobin dẫn đến giảm khả năng vận chuyển oxy của máu, gây tình trạng thiếu oxy mô.",
        "potential_causes": "Thiếu sắt (mạn tính) hoặc Mất máu cấp tính/rỉ rả hoặc Tan máu (Hemolysis) hoặc Rối loạn chức năng tủy xương.",
        "scientific_evidence": {
            "source": "World Health Organization (WHO), 2011",
            "quote": "Thiếu máu được xác định khi Hemoglobin < 13.0 g/dL ở nam giới và < 12.0 g/dL ở nữ giới.",
            "url": "https://www.who.int/vmnis/anaemia/en/"
        },
        "key_indicators": ["HGB", "RBC", "HCT", "Mentzer_Index"]
    },
    "Bleeding": {
        "title": "Nguy cơ Xuất huyết (Thrombocytopenia)",
        "mechanism": "Giảm số lượng tiểu cầu làm suy yếu nút chặn cầm máu sơ cấp, kéo dài thời gian chảy máu.",
        "potential_causes": "Sốt xuất huyết Dengue (giai đoạn nguy hiểm) hoặc Giảm tiểu cầu tự miễn hoặc Xơ gan cường lách hoặc Suy tủy.",
        "scientific_evidence": {
            "source": "American Society of Hematology (ASH)",
            "quote": "Số lượng tiểu cầu < 150 K/uL là ngưỡng bắt đầu của tình trạng giảm tiểu cầu, tăng nguy cơ xuất huyết.",
            "url": "https://www.hematology.org/"
        },
        "key_indicators": ["PLT"]
    },
    "Deficiency": {
        "title": "Rối loạn Hình thái & Thiếu hụt Vi chất",
        "mechanism": "Bất thường trong kích thước hồng cầu phản ánh sự thiếu hụt nguyên liệu (Sắt, B12) trong quá trình tạo máu.",
        "potential_causes": "Thiếu Vitamin B12/Folate (Hồng cầu to) hoặc Thiếu sắt giai đoạn sớm (Hồng cầu nhỏ) hoặc Bệnh lý Thalassemia.",
        "scientific_evidence": {
            "source": "American Family Physician (AFP), 2010",
            "quote": "Chỉ số MCV (ngưỡng 80-100 fL) là tiêu chuẩn vàng để phân loại căn nguyên thiếu máu theo hình thái.",
            "url": "https://www.aafp.org/afp/2010/1115/p1117.html"
        },
        "key_indicators": ["MCV", "MCH", "RDW", "Mentzer_Index"]
    }
}

def start_training():
    print("--- [BẮT ĐẦU] HUẤN LUYỆN AI VỚI 1600+ HỒ SƠ BỆNH ÁN ---")
    
    # 1. Nạp dữ liệu
    df = pd.read_csv('MASTER_AI_DATASET_COMBINED.csv')
    
    # 2. Feature Engineering (Tính toán các tỷ lệ vàng)
    print(">> Đang tối ưu hóa các đặc trưng lâm sàng (Mentzer, NLR, PLR)...")
    df['Mentzer_Index'] = df['MCV'] / (df['RBC'] + 0.0001)
    df['NLR'] = df['Neutrophils'] / (df['Lymphocytes'] + 0.0001)
    df['PLR'] = df['PLT'] / (df['Lymphocytes'] + 0.0001)

    # 3. Xử lý nhãn (Targets)
    targets = ['Infection', 'Anemia', 'Bleeding', 'Deficiency']
    for t in targets:
        df[t] = df['Logic_Keywords'].apply(lambda x: 1 if str(t) in str(x) else 0)

    features = ['HCT', 'HGB', 'Lymphocytes', 'MCH', 'MCHC', 'MCV', 'Monocytes', 
                'Neutrophils', 'PLT', 'RBC', 'RDW', 'WBC', 'Gender', 'Age', 
                'Mentzer_Index', 'NLR', 'PLR']

    X = df[features]
    y = df[targets]

    # Chia tập dữ liệu để kiểm tra (Test set 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Huấn luyện thuật toán Random Forest
    print(f">> Đang huấn luyện Random Forest trên {len(X_train)} mẫu...")
    model = RandomForestClassifier(n_estimators=250, max_depth=15, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 5. Đánh giá chuyên sâu (Evaluation)
    print("\n" + "="*40)
    print("BẢNG ĐIỂM CHẤT LƯỢNG AI (CLASSIFICATION REPORT)")
    print("="*40)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=targets, zero_division=0))

    # 6. Vẽ biểu đồ Feature Importance (Cực kỳ quan trọng để giải thích)
    importances = model.feature_importances_
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.title('Tầm quan trọng của các chỉ số trong quyết định của AI')
    plt.xlabel('Trọng số ảnh hưởng')
    plt.tight_layout()
    plt.savefig('clinical_feature_importance.png')
    print(">> Đã lưu biểu đồ: clinical_feature_importance.png")

    # 7. Đóng gói bộ não AI
    payload = {
        'model': model,
        'features': features,
        'knowledge': CLINICAL_KNOWLEDGE,
        'stats': {'accuracy': model.score(X_test, y_test)}
    }
    joblib.dump(payload, 'blood_ai_ultimate.pkl')
    
    # 8. Xuất file JSON dẫn chứng riêng để dự phòng
    with open('scientific_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(CLINICAL_KNOWLEDGE, f, ensure_ascii=False, indent=4)

    print("\n>> HOÀN TẤT! Mô hình đã sẵn sàng cho run.py.")

if __name__ == "__main__":
    start_training()