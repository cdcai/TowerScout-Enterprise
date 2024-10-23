#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#
from azure.identity import ClientSecretCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class devSecrets:
    def getSecret(secret_name):
     
    #  SP info to access Key Vault
     tenant_id = '9ce70869-60db-44fd-abe8-d2767077fc8f'
     client_id = '53807466-0da1-4622-8e24-096d55bd3f3e'
     client_secret = 'wxK8Q~rwCLgCZmkVb7qgvV3wLBYmMSDOCzCFPbLO'
     key_vault_url = "https://cselstowerscrtdevkv01.vault.azure.net/"

     # Initialize the credential
     credential = DefaultAzureCredential()
     # Get the access token for the Key Vault
     credential = ClientSecretCredential(tenant_id, client_id, client_secret)
     # token = credential.get_token("https://management.azure.com/.default")
     # Create a SecretClient
     client = SecretClient(vault_url=key_vault_url, credential=credential)
    #  print(client.get_secret(secret_name).value)
     return client.get_secret(secret_name).value

class prodSecrets:
   def getSecret(secret_name):
   
     # Initialize the credential
     credential = DefaultAzureCredential()
     key_vault_url = "https://cselstowerscrtprdkv01.vault.azure.net/"
     # Create a SecretClient instance using the key vault URL and credential
     client = SecretClient(vault_url=key_vault_url, credential=credential)
     return client.get_secret(secret_name).value

class localSecrets:
   def getSecret(secret_name):
   

     # Initialize the credential
     credential = DefaultAzureCredential()
     key_vault_url = "https://towerscout-mapkeyvault.vault.azure.net/"
     # Create a SecretClient instance using the key vault URL and credential
     client = SecretClient(vault_url=key_vault_url, credential=credential)
     return client.get_secret(secret_name).value
