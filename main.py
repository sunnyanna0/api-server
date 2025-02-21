from fastapi import FastAPI
import model

# FastAPI 서버 생성
app = FastAPI()

# 모델 인스턴스 생성 (AND, OR, NOT)
and_model = model.AndModel()
or_model = model.OrModel()
not_model = model.NOTModel()
and_model.train()
or_model.train()
not_model.train()

# 기본 엔드포인트
@app.get("/")
def read_root():
    return {"message": "논리연산 AI 서버입니다. /predict/{operation}/{left}/{right} 경로로 요청하세요."}

# 논리 연산 예측 API
@app.get("/predict/{operation}/{left}")
@app.get("/predict/{operation}/{left}/{right}")
def predict(operation: str, left: int, right: int = None):
    if operation == "AND":
        result = and_model.predict([left, right])
    elif operation == "OR":
        result = or_model.predict([left, right])
    elif operation == "NOT":
        if right is not None:
            return {"error": "NOT 연산에는 하나의 입력값만 필요합니다."}
        result = not_model.predict([left])
    else:
        return {"error": "지원하지 않는 연산입니다. (AND, OR, NOT만 가능)"}

    return {"operation": operation, "left": left, "right": right, "result": result}

# 논리 연산 모델 학습 API (POST 요청)
@app.post("/train/{operation}")
def train(operation: str):
    if operation == "AND":
        and_model.train()
    elif operation == "OR":
        or_model.train()
    elif operation == "NOT":
        not_model.train()
    else:
        return {"error": "지원하지 않는 연산입니다. (AND, OR, NOT만 가능)"}

    return {"operation": operation, "result": "Training complete"}
