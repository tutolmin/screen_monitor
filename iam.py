import yandexcloud

from yandex.cloud.iam.v1.iam_token_service_pb2 import (CreateIamTokenRequest)
from yandex.cloud.iam.v1.iam_token_service_pb2_grpc import IamTokenServiceStub

key_path = '<JSON-файл_c_ключами>'

# Чтение закрытого ключа из JSON-файла
with open(key_path, 'r') as f:
  obj = f.read() 
  obj = json.loads(obj)
  private_key = obj['private_key']
  key_id = obj['id']
  service_account_id = obj['service_account_id']

sa_key = {
    "id": key_id,
    "service_account_id": service_account_id,
    "private_key": private_key
}

def create_iam_token():
  jwt = create_jwt()
  
  sdk = yandexcloud.SDK(service_account_key=sa_key)
  iam_service = sdk.client(IamTokenServiceStub)
  iam_token = iam_service.Create(
      CreateIamTokenRequest(jwt=jwt)
  )
  
  print("Your iam token:")
  print(iam_token.iam_token)

  return iam_token.iam_token
