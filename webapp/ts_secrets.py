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
import json
import os


class devSecrets:
    def getSecret(secret_name):
     
    #  SP info to access Key Vault
     
     # Read the json config file
     with open('config.development.json', 'r') as file:
       data = json.load(file)  # Load the JSON data into a Python dictionary

       tenant_id = data['tenant_id']
       client_id = data['client_id']
       client_secret = data['client_secret']
       key_vault_url = data['key_vault_url'] 
     # Get the access token for the Key Vault
     credential = ClientSecretCredential(tenant_id, client_id, client_secret)
     # token = credential.get_token("https://management.azure.com/.default")
     # Create a SecretClient
     client = SecretClient(vault_url=key_vault_url, credential=credential)
     # print(client.get_secret(secret_name).value)
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
