import os
from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# 强制使用脚本所在目录加载模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# 加载模型和 scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"成功加载模型: {model_path}")
    print(f"成功加载标准化器: {scaler_path}")
except Exception as e:
    print(f"加载失败: {e}")
    exit()

FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>加州房价预测器</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; background: #f8f9fa; padding: 20px; }
        h1 { color: #343a40; text-align: center; }
        form { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        label { display: block; margin: 15px 0 5px; font-weight: bold; color: #495057; }
        input { width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; box-sizing: border-box; }
        button { margin-top: 30px; width: 100%; padding: 14px; background: #007bff; color: white; border: none; border-radius: 6px; font-size: 18px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 30px; padding: 20px; background: #d4edda; border-radius: 8px; text-align: center; font-size: 24px; color: #155724; }
        .error { margin-top: 30px; padding: 20px; background: #f8d7da; border-radius: 8px; text-align: center; color: #721c24; }
    </style>
</head>
<body>
    <h1>🏠 加州房价预测器（脏数据处理版）</h1>
    <form method="post">
        <label>家庭收入中位数 (MedInc):</label>
        <input type="number" step="0.01" name="MedInc" value="4.0" required>

        <label>房屋年龄 (HouseAge):</label>
        <input type="number" name="HouseAge" value="20" required>

        <label>平均房间数 (AveRooms):</label>
        <input type="number" step="0.01" name="AveRooms" value="6.0" required>

        <label>平均卧室数 (AveBedrms):</label>
        <input type="number" step="0.01" name="AveBedrms" value="1.0" required>

        <label>人口数 (Population):</label>
        <input type="number" name="Population" value="1000" required>

        <label>平均入住率 (AveOccup):</label>
        <input type="number" step="0.01" name="AveOccup" value="3.0" required>

        <label>纬度 (Latitude):</label>
        <input type="number" step="0.01" name="Latitude" value="34.0" required>

        <label>经度 (Longitude):</label>
        <input type="number" step="0.01" name="Longitude" value="-118.0" required>

        <button type="submit">立即预测房价</button>
    </form>

    {% if prediction is not none %}
        {% if prediction is string and '错误' in prediction %}
            <div class="error">
                <h2>{{ prediction }}</h2>
            </div>
        {% else %}
            <div class="result">
                <h2>预测房价：${{ "{:,.0f}".format(prediction) }}</h2>
                <p>（单位：美元，基于 California Housing 数据集）</p>
            </div>
        {% endif %}
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            data = {feat: float(request.form[feat]) for feat in FEATURES}
            df = pd.DataFrame([data])
            X_scaled = scaler.transform(df.values)
            pred = model.predict(X_scaled)[0]
            prediction = pred * 100000  # 转为美元
        except Exception as e:
            prediction = f"输入错误: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)