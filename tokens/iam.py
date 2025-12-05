import yandexcloud

from yandex.cloud.iam.v1.iam_token_service_pb2 import (CreateIamTokenRequest)
from yandex.cloud.iam.v1.iam_token_service_pb2_grpc import IamTokenServiceStub

import time
import jwt
import json

key_path = '../keys/authorized_key.json'

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

def create_jwt():
    now = int(time.time())
    payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': service_account_id,
            'iat': now,
            'exp': now + 3600
        }

    # Формирование JWT.
    encoded_token = jwt.encode(
        payload,
        private_key,
        algorithm='PS256',
        headers={'kid': key_id}
    )

    print(encoded_token)

    return encoded_token

#create_jwt()
create_iam_token()
